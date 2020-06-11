import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time
from random import sample as rsample
from random import seed
from data_utils import load_task

random_seed = 2
embedding_dim = 100
n_memories = 20
gradient_clip_value = 40
batch_size = 32
tie_keys = True
learn_keys = True
tasks = [1]
data_dir = "data/tasks_1-20_v1-2/en-10k"
STATE_PATH = './trained_models/task_{}.pth'
OPTIM_PATH = './trained_models/task_{}.pth'

# for reproducibility
torch.manual_seed(random_seed)
np.random.seed(random_seed)
seed(random_seed)


def print_start_train_message(task):
    key_state_txt = "tied to vocab" if tie_keys else "NOT tied to vocab"
    key_learned_txt = "learned" if learn_keys else "NOT learned"
    print("start learning task {}\n".format(task) +
          "random seed is {}\n".format(random_seed) +
          "embedding dimension is {}\n".format(embedding_dim) +
          "number of memories is {}\n".format(n_memories) +
          "gradient clip value is {}\n".format(gradient_clip_value) +
          "batch size is {}\n".format(batch_size) +
          "keys are {}\n".format(key_state_txt) +
          "keys are {}\n".format(key_learned_txt))


def print_start_test_message(task):
    print("testing task {}\n".format(task) +
          "random seed is {}\n".format(random_seed) +
          "embedding dimension is {}\n".format(embedding_dim) +
          "number of memories is {}\n".format(n_memories))


def get_vocab(train, test):
    vocab = set()
    samples = train + test
    for story, query, answer in samples:
        for word in [word for sentence in story for word in sentence] + query + answer:
            vocab.add(word)
    vocab = list(vocab)
    vocab.sort()
    return vocab, len(vocab) + 1


def init_embedding_matrix(vocab, device):
    token_to_idx = {token: torch.tensor(i+1) for i, token in enumerate(vocab)}
    embeddings_matrix = nn.Embedding(len(vocab) + 1, embedding_dim, 0)
    cuda_embeddings_matrix = nn.Embedding(len(vocab) + 1, embedding_dim, 0).to(device)

    return cuda_embeddings_matrix, embeddings_matrix, token_to_idx


def get_len_max_sentence_and_story(data):
    len_max_sentence = 0
    len_max_story = 0
    for story, query, answer in data:
        if len(story) > len_max_story:
            len_max_story = len(story)
        for sentence in story:
            if len(sentence) > len_max_sentence:
                len_max_sentence = len(sentence)
        if len(query) > len_max_sentence:
            len_max_sentence = len(query)
    return len_max_sentence, len_max_story


def vectorize_data(data, token_to_idx, embeddings_matrix, len_max_sentence, len_max_story):
    vec_stories = torch.zeros((len(data), len_max_story, len_max_sentence, embedding_dim), requires_grad=False)
    vec_queries = torch.zeros((len(data), len_max_sentence, embedding_dim), requires_grad=False)
    vec_answers = torch.zeros((len(data)), requires_grad=False, dtype=torch.long)

    i = 0
    for story, query, answer in data:
        vec_curr_story = torch.zeros((len_max_story, len_max_sentence, embedding_dim), requires_grad=False)
        for j, sentence in enumerate(story):
            word_padding_size = max(0, len_max_sentence - len(sentence))
            vec_curr_story[j] = embeddings_matrix(torch.tensor([token_to_idx[w] for w in sentence] + [0] * word_padding_size))

        sentence_padding_size = max(0, len_max_story - len(story))
        for j in range(1, sentence_padding_size + 1):
            vec_curr_story[-j] = embeddings_matrix(torch.tensor([0] * len_max_sentence))

        vec_stories[i] = vec_curr_story

        word_padding_size = max(0, len_max_sentence - len(query))
        vec_curr_query = embeddings_matrix(torch.tensor([token_to_idx[w] for w in query] + [0] * word_padding_size))
        vec_queries[i] = vec_curr_query

        # vec_answers[i] = torch.tensor(token_to_idx[answer[0]], requires_grad=False, dtype=torch.long)
        vec_answers[i] = token_to_idx[answer[0]].clone().detach()

        i += 1

    return vec_stories, vec_queries, vec_answers


def batch_generator(data, batch_size):
    vec_stories, vec_queries, vec_answers = data
    len_data = len(vec_stories)

    perm = torch.randperm(len_data)
    # vec_stories, vec_queries, vec_answers = vec_stories[perm], vec_queries[perm], vec_answers[perm]
    pos = 0
    while pos < len_data:
        if pos < len_data - batch_size:
            yield vec_stories[perm[pos:pos + batch_size]], vec_queries[perm[pos:pos + batch_size]], vec_answers[perm[pos:pos + batch_size]]
            pos = pos + batch_size
        else:
            return vec_stories[perm[pos:]], vec_queries[perm[pos:]], vec_answers[perm[pos:]]


def get_key_tensors(vocab, embeddings_matrix, token_to_idx, device,  tied=True, learned=True):
    """
    returns a list of key tensors with length n_memories
    list may be randomly initialized (current version) or tied to specific entities
    """
    if tied:
        keys = torch.zeros((n_memories, embedding_dim), device=device)
        for i, word in enumerate(vocab):
            if i < n_memories:
                # keys[i] = embeddings_matrix(torch.tensor(token_to_idx[word], device=device))
                keys[i] = embeddings_matrix(token_to_idx[word].clone().detach().to(device))
        return nn.Parameter(keys, requires_grad=True).to(device) if learned else keys

    mean = torch.zeros((n_memories, embedding_dim), device=device)
    standard_deviation = torch.full((n_memories, embedding_dim), 0.1, device=device)
    keys = torch.normal(mean, standard_deviation)
    return nn.Parameter(keys, requires_grad=True).to(device) if learned else keys


def get_matrix_weights(device):
    """
    :return: initial weights for any og the matrices U, V, W
     weights may be randomly initialized (current version) or initialized to zeros or the identity matrix
    """
    init_mean = torch.zeros((embedding_dim, embedding_dim), device=device)
    init_standard_deviation = torch.full((embedding_dim, embedding_dim), 0.1, device=device)

    return nn.Parameter(torch.normal(init_mean, init_standard_deviation)).to(device)


def get_r_matrix_weights(vocab_size, device):
    """
    :return: initial weights for any og the matrices U, V, W
     weights may be randomly initialized (current version) or initialized to zeros or the identity matrix
    """
    init_mean = torch.zeros((vocab_size, embedding_dim), device=device)
    init_standard_deviation = torch.full((vocab_size, embedding_dim), 0.1, device=device)

    return nn.Parameter(torch.normal(init_mean, init_standard_deviation)).to(device)


def get_non_linearity():
    """
    :return: the non-linearity function to be used in the model.
    this may be a parametric ReLU (current version) or (despite its name) the identity
    """
    # return nn.PReLU(num_parameters=embedding_dim, init=1)
    return nn.PReLU(init=1)


# batch training
class EntNet(nn.Module):
    def __init__(self, vocab_size, keys, len_max_sentence, device):
        super(EntNet, self).__init__()
        self.len_max_sentence = len_max_sentence
        self.device = device

        # Encoder
        self.input_encoder_multiplier = nn.Parameter(torch.ones((len_max_sentence, embedding_dim), device=device)).to(device)
        self.query_encoder_multiplier = nn.Parameter(torch.ones((len_max_sentence, embedding_dim), device=device)).to(device)
        # self.query_encoder_multiplier = self.input_encoder_multiplier

        # Memory
        self.keys = keys
        self.memories = None

        # self.gates = nn.Parameter(torch.zeros(n_memories), requires_grad=True)

        self.U = nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)
        self.V = nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)
        self.W = nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)
        self.U.weight = get_matrix_weights(device)
        self.V.weight = get_matrix_weights(device)
        self.W.weight = get_matrix_weights(device)

        self.in_non_linearity = get_non_linearity().to(device)
        # self.query_non_linearity = get_non_linearity().to(device)
        self.query_non_linearity = self.in_non_linearity

        # Decoder
        self.R = nn.Linear(vocab_size, embedding_dim, bias=False).to(device)
        self.H = nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)
        self.R.weight = get_r_matrix_weights(vocab_size, device)
        self.H.weight = get_matrix_weights(device)

    def init_new_memories(self, device, batch_size):
        # self.memories = torch.tensor(self.keys, requires_grad=False, device=device).repeat(batch_size, 1, 1)
        self.memories = self.keys.clone().detach().to(device).repeat(batch_size, 1, 1)

    def forward(self, batch):

        # re-initialize memories to key-values
        self.init_new_memories(self.device, len(batch))

        # Encoder
        batch = batch * self.input_encoder_multiplier
        batch = batch.sum(dim=2)

        # Memory
        for sentence_idx in range(batch.shape[1]):
            sentence = batch[:, sentence_idx]
            sentence_memory_repeat = sentence.repeat(1, n_memories).view(batch_size, n_memories, -1)

            memory_gate = (sentence * self.memories.permute(1, 0, 2)).permute(1, 0, 2).sum(dim=2)
            key_gate = (sentence_memory_repeat * self.keys).sum(dim=2)
            gate = torch.sigmoid(memory_gate + key_gate)

            update_candidate = self.in_non_linearity(self.U(self.memories) + self.V(self.keys) + self.W(sentence_memory_repeat))
            self.memories = self.memories + (update_candidate.permute(2, 0, 1) * gate).permute(1, 2, 0)
            self.memories = (self.memories.permute(2, 0, 1) / torch.norm(self.memories, dim=2)).permute(1, 2, 0)

    def decode(self, batch):
        # Decoder
        # query = query.to(device)
        batch = batch * self.query_encoder_multiplier
        batch = batch.sum(dim=1)
        answers_probabilities = F.softmax((batch * self.memories.permute(1, 0, 2)).sum(dim=2).t(), dim=0)
        scores = (self.memories.permute(2, 0 ,1) * answers_probabilities).permute(1, 2 ,0).sum(dim=1)
        results = self.R(self.query_non_linearity(batch + self.H(scores)))
        return results


def train(task, device):
    print_start_train_message(task)

    train, test = load_task(data_dir, task)

    vocab, vocab_size = get_vocab(train, test)
    cuda_embeddings_matrix, embeddings_matrix, token_to_idx = init_embedding_matrix(vocab, device)
    keys = get_key_tensors(vocab, cuda_embeddings_matrix, token_to_idx, device, tie_keys, learn_keys)

    len_max_sentence, len_max_story = get_len_max_sentence_and_story(train + test)
    vec_train = vectorize_data(train, token_to_idx, embeddings_matrix, len_max_sentence, len_max_story)

    entnet = EntNet(vocab_size, keys, len_max_sentence, device)
    entnet.to(device)
    entnet = entnet.float()
    # entnet.load_state_dict(torch.load(STATE_PATH.format(task, 0)))

    ##### Define Loss and Optimizer #####
    criterion = nn.CrossEntropyLoss().to(device)
    learning_rate = 0.01
    optimizer = optim.Adam(entnet.parameters(), lr=learning_rate)
    # optimizer.load_state_dict(torch.load(OPTIM_PATH.format(task, 0)))

    ##### Train Model #####
    epoch = 0
    max_stuck_epochs = 10
    epsilon = 0.01
    loss_history = [np.inf] * max_stuck_epochs
    net_history = [None] * max_stuck_epochs
    optim_history = [None] * max_stuck_epochs

    while True:
        epoch_loss = 0.0
        running_loss = 0.0
        correct = 0
        start_time = time.time()
        for i, batch in enumerate(batch_generator(vec_train, batch_size)):
            batch_stories, batch_queries, batch_answers = batch
            # batch_stories, batch_queries, batch_answers = torch.tensor(batch_stories, requires_grad=False, device=device),\
            #                                               torch.tensor(batch_queries, requires_grad=False, device=device),\
            #                                               torch.tensor(batch_answers, requires_grad=False, device=device)

            batch_stories, batch_queries, batch_answers = batch_stories.clone().detach().to(device), \
                                                          batch_queries.clone().detach().to(device), \
                                                          batch_answers.clone().detach().to(device)

            entnet(batch_stories)
            output = entnet.decode(batch_queries)
            loss = criterion(output, batch_answers)
            loss.backward()

            running_loss += loss.item()
            epoch_loss += loss.item()

            # nn.utils.clip_grad_value_(entnet.parameters(), gradient_clip_value)
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()

            pred_idx = np.argmax(output.cpu().detach().numpy(), axis=1)
            for j in range(len(output)):
                if pred_idx[j] == batch_answers[j].item():
                    correct += 1

            if i % 50 == 49:
                # print statistics
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

                print('[%d, %5d] correct: %d out of %d' % (epoch + 1, i + 1, correct, 50 * batch_size))
                correct = 0

        # very loose approximation for the average loss over the epoch
        epoch_loss = epoch_loss / (len(vec_train[0]) / batch_size)
        # print epoch time
        end_time = time.time()
        print("###################################################################################################")
        print(end_time - start_time)
        print('epoch loss: %.3f' % epoch_loss)
        print("###################################################################################################")

        ##### Save Trained Model #####
        # torch.save(entnet.state_dict(), STATE_PATH.format(task, epoch))
        # torch.save(optimizer.state_dict(), OPTIM_PATH.format(task, epoch))

        net_history.append(entnet.state_dict())
        optim_history.append(optimizer.state_dict())
        net_history = net_history[1:]
        optim_history = optim_history[1:]

        loss_history.append(epoch_loss)
        loss_history = loss_history[1:]
        if loss_history[0] - min(loss_history[1:]) < epsilon:
            torch.save(net_history[-1], STATE_PATH.format(task))
            torch.save(optim_history[-1], OPTIM_PATH.format(task))
            break

        # adjust learning rate every 25 epochs until 200 epochs
        if epoch < 200 and epoch % 25 == 24:
            learning_rate = learning_rate / 2
            optimizer = optim.Adam(entnet.parameters(), lr=learning_rate)
        if epoch == 200:
            torch.save(net_history[-1], STATE_PATH.format(task))
            torch.save(optim_history[-1], OPTIM_PATH.format(task))
            break

        epoch += 1

    print('Finished Training')


def test(task):
    print_start_test_message(task)

    train, test = load_task(data_dir, task)

    vocab, vocab_size = get_vocab(train, test)
    embeddings_matrix, token_to_idx = init_embedding_matrix(vocab)
    keys = get_key_tensors(vocab, embeddings_matrix, token_to_idx, tie_keys, learn_keys)

    n_input_words = get_len_max_sentence_and_story(train + test)
    vec_test = vectorize_data(test, token_to_idx, embeddings_matrix, n_input_words)

    entnet = EntNet(vocab_size, keys, n_input_words)
    entnet.load_state_dict(torch.load(STATE_PATH.format(task, 34)))

    ##### Define Loss and Optimizer #####
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        correct = 0
        start_time = time.time()
        for i, sample in enumerate(vec_test):
            # get the inputs; data is a list of [inputs, labels]
            story, query, answer = sample

            entnet(story)
            output = entnet.decode(query)
            loss = criterion(output, answer)

            pred_idx = np.argmax(output.detach().numpy())
            if pred_idx == answer[0].item():
                correct += 1
            if i % 500 == 499:  # print every 500 samples
                print('%5d correct: %d' % (i + 1, correct))

        # print epoch time
        end_time = time.time()
        print("###################################################################################################")
        print(end_time - start_time)
        print("###################################################################################################")

        print("got {} correct out of {} samples\n".format(correct, len(vec_test)) +
              "finished testing\n")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    for task in tasks:
        train(task, device)
    # for task in tasks:
    #     test(task, device)


if __name__ == "__main__":
    main()