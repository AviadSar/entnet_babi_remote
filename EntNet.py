import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nltk
import copy
import time
from random import sample as rsample
from random import seed
from data_utils import load_task, get_tokenized_text
import argparse
import os

verbose = False
random_seed = 2
embedding_dim = 100
n_memories = 20
gradient_clip_value = 40
batch_size = 32
tie_keys = True
learn_keys = True
max_stuck_epochs = 6
teach = False
min_improvement = 0.0
n_tries = 10
optimizer_name = 'adam'
tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# entnet_threshold = [0] * 20
entnet_threshold = [0, 0.1, 4.1, 0, 0.3, 0.2, 0, 0.5, 0.1, 0.6, 0.3, 0, 1.3, 0, 0, 0.2, 0.5, 0.3, 2.3, 0]
data_dir = "data/tasks_1-20_v1-2/en-10k"
STATE_PATH = './trained_models/task_{}.pth'
OPTIM_PATH = './trained_models/task_{}.pth'
cuda = True


def print_start_train_message(task):
    key_state_text = "tied to vocab" if tie_keys else "NOT tied to vocab"
    key_learned_text = "learned" if learn_keys else "NOT learned"
    cuda_text = "gpu" if cuda else "cpu"
    verbose_text = "verbose" if verbose else "non-verbose"
    teaching_text = "enabled" if teach else "disabled"
    print("start learning task {}\n".format(task) +
          "learning on {}\n".format(cuda_text) +
          "random seed is {}\n".format(random_seed) +
          "embedding dimension is {}\n".format(embedding_dim) +
          "number of memories is {}\n".format(n_memories) +
          "gradient clip value is {}\n".format(gradient_clip_value) +
          "maximum stuck epochs is {}\n".format(max_stuck_epochs) +
          "teaching is {}\n".format(teaching_text) +
          "minimal improvement is {}\n".format(min_improvement) +
          "batch size is {}\n".format(batch_size) +
          "keys are {}\n".format(key_state_text) +
          "keys are {}\n".format(key_learned_text) +
          "{} mode\n".format(verbose_text))


def print_start_test_message(task):
    print("testing task {}\n".format(task))
    if verbose:
        print("random seed is {}\n".format(random_seed) +
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


def get_vocab_and_entities(task):
    is_noun = lambda pos: pos[:2] == 'NN'

    vocab = set()
    entities = set()

    try:
        nltk.pos_tag("")
    except:
        nltk.download('averaged_perceptron_tagger')

    text = get_tokenized_text(data_dir, task)

    for word in text:
        if word.isalpha():
            vocab.add(word)

    vocab = list(vocab)
    vocab.sort()

    for (word, pos) in nltk.pos_tag(text):
        if is_noun(pos):
            entities.add(word)

    entities = list(entities)
    entities.sort()

    return vocab, len(vocab) + 1, entities


def init_embedding_matrix(vocab, device):
    # token_to_idx = {token: torch.tensor(i+1) for i, token in enumerate(vocab)}
    token_to_idx = {token: i+1 for i, token in enumerate(vocab)}
    embeddings_matrix = nn.Embedding(len(vocab) + 1, embedding_dim, 0).to(device)

    init_mean = torch.zeros((len(vocab) + 1, embedding_dim), device=device)
    init_standard_deviation = torch.cat((torch.full((1, embedding_dim), 0, device=device), torch.full((len(vocab), embedding_dim), 0.1, device=device)))

    embeddings_matrix.weight = nn.Parameter(torch.normal(init_mean, init_standard_deviation), requires_grad=True).to(device)

    return embeddings_matrix, token_to_idx


def get_len_max_story(data):
    len_max_story = 0
    for story, query, answer in data:
        if len(story) > len_max_story:
            len_max_story = len(story)
    return len_max_story
    # return np.max([len(tuple[0]) for tuple in batch])


def get_len_max_sentence(data):
    len_max_sentence = 0
    for story, query, answer in data:
        for sentence in story:
            if len(sentence) > len_max_sentence:
                len_max_sentence = len(sentence)
        if len(query) > len_max_sentence:
            len_max_sentence = len(query)
    return len_max_sentence


def vectorize_data(data, len_max_sentence, len_max_story):
    vec_stories = torch.zeros((len(data), len_max_story, len_max_sentence), dtype=torch.long, requires_grad=False)
    vec_queries = torch.zeros((len(data), len_max_sentence), dtype=torch.long, requires_grad=False)
    vec_answers = torch.zeros((len(data)), requires_grad=False, dtype=torch.long)

    i = 0
    for story, query, answer in data:
        vec_curr_story = torch.zeros((len_max_story, len_max_sentence), requires_grad=False)
        for j, sentence in enumerate(story):
            word_padding_size = max(0, len_max_sentence - len(sentence))
            vec_curr_story[j] = torch.tensor(sentence + [0] * word_padding_size)

        sentence_padding_size = max(0, len_max_story - len(story))
        for j in range(1, sentence_padding_size + 1):
            vec_curr_story[-j] = torch.tensor([0] * len_max_sentence)

        vec_stories[i] = vec_curr_story

        word_padding_size = max(0, len_max_sentence - len(query))
        vec_curr_query = torch.tensor(query + [0] * word_padding_size)
        vec_queries[i] = vec_curr_query

        vec_answers[i] = torch.tensor(answer)

        i += 1

    return vec_stories, vec_queries, vec_answers


def indexize_data(data, token_to_idx, len_max_sentence):
    indexed_data = []

    for story, query, answer in data:
        indexed_story = []
        for sentence in story:
            word_padding_size = max(0, len_max_sentence - len(sentence))
            indexed_story.append([token_to_idx[w] for w in sentence] + [0] * word_padding_size)


        word_padding_size = max(0, len_max_sentence - len(query))
        indexed_query = [token_to_idx[w] for w in query] + [0] * word_padding_size

        indexed_answer = token_to_idx[answer[0]]

        indexed_data.append((indexed_story, indexed_query, indexed_answer))

    indexed_data.sort(key=lambda tuple: len(tuple[0]))
    return indexed_data


def batch_generator(data, len_max_sentence, batch_size, permute='full'):
    len_data = len(data)

    if permute == 'half':
        last_batch_pos = int(np.ceil(len_data / batch_size))
        perm = np.random.permutation(last_batch_pos)

        for pos in perm:
            if pos != last_batch_pos - 1:
                batch = data[pos * batch_size:(pos + 1) * batch_size]
                len_max_story = len(batch[-1][0])
                vec_batch = vectorize_data(batch, len_max_sentence, len_max_story)
                yield vec_batch
            else:
                batch = data[pos * batch_size:]
                len_max_story = len(batch[-1][0])
                vec_batch = vectorize_data(batch, len_max_sentence, len_max_story)
                yield vec_batch
        return

    perm_data = data # permute argument if 'no', of anything that's not 'half' or 'full'
    if permute == 'full':
        perm_data = np.random.permutation(np.array(data, dtype=object))

    pos = 0
    while pos < len_data:
        if pos < len_data - batch_size:
            batch = perm_data[pos:pos + batch_size]
            len_max_story = get_len_max_story(batch)
            vec_batch = vectorize_data(batch, len_max_sentence, len_max_story)
            yield vec_batch
            pos = pos + batch_size
        else:
            batch = perm_data[pos:]
            len_max_story = get_len_max_story(batch)
            vec_batch = vectorize_data(batch, len_max_sentence, len_max_story)
            yield vec_batch
            return


def get_key_tensors(entities, embeddings_matrix, token_to_idx, device,  tied=True):
    """
    returns a list of key tensors with length n_memories
    list may be randomly initialized (current version) or tied to specific entities
    """
    mean = torch.zeros((n_memories, embedding_dim), device=device)
    standard_deviation = torch.full((n_memories, embedding_dim), 0.1, device=device)
    keys = torch.normal(mean, standard_deviation)

    if tied:
        # keys = torch.zeros(n_memories, dtype=torch.long, device=device)
        # for i, word in enumerate(vocab):
        for i, word in enumerate(entities):
            if i < n_memories:
                # keys[i] = embeddings_matrix(torch.tensor(token_to_idx[word], device=device))
                keys[i] = embeddings_matrix(torch.tensor(token_to_idx[word]).to(device))
        return nn.Parameter(keys, requires_grad=False).to(device)

    return nn.Parameter(keys, requires_grad=False).to(device)


def get_matrix_weights(device):
    """
    :return: initial weights for any og the matrices U, V, W
     weights may be randomly initialized (current version) or initialized to zeros or the identity matrix
    """
    init_mean = torch.zeros((embedding_dim, embedding_dim), device=device)
    init_standard_deviation = torch.full((embedding_dim, embedding_dim), 0.1, device=device)

    return nn.Parameter(torch.normal(init_mean, init_standard_deviation), requires_grad=True).to(device)


def get_r_matrix_weights(vocab_size, device):
    """
    :return: initial weights for any og the matrices U, V, W
     weights may be randomly initialized (current version) or initialized to zeros or the identity matrix
    """
    init_mean = torch.zeros((vocab_size, embedding_dim), device=device)
    init_standard_deviation = torch.full((vocab_size, embedding_dim), 0.1, device=device)

    return nn.Parameter(torch.normal(init_mean, init_standard_deviation), requires_grad=True).to(device)


def get_non_linearity():
    """
    :return: the non-linearity function to be used in the model.
    this may be a parametric ReLU (current version) or (despite its name) the identity
    """
    # return nn.PReLU(num_parameters=embedding_dim, init=1)
    return nn.PReLU(init=1)


# batch training
class EntNet(nn.Module):
    def __init__(self, vocab_size, keys, len_max_sentence, embeddings_matrix, device):
        super(EntNet, self).__init__()
        self.len_max_sentence = len_max_sentence
        self.device = device

        #embedding
        self.embedding_matrix = embeddings_matrix

        # Encoder
        self.input_encoder_multiplier = nn.Parameter(torch.ones((len_max_sentence, embedding_dim), device=device), requires_grad=True).to(device)
        self.query_encoder_multiplier = nn.Parameter(torch.ones((len_max_sentence, embedding_dim), device=device), requires_grad=True).to(device)
        # self.query_encoder_multiplier = self.input_encoder_multiplier

        # Memory
        self.keys = keys
        self.embedded_keys = self.keys
        if learn_keys:
            self.embedded_keys = nn.Parameter(self.embedded_keys, requires_grad=True)
        self.memories = None

        # self.gates = nn.Parameter(torch.zeros(n_memories), requires_grad=True)

        self.U = nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)
        self.V = nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)
        self.W = nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)
        # self.U.weight = nn.Parameter(get_matrix_weights(device), requires_grad=True).to(device)
        # self.V.weight = nn.Parameter(get_matrix_weights(device), requires_grad=True).to(device)
        # self.W.weight = nn.Parameter(get_matrix_weights(device), requires_grad=True).to(device)
        self.U.weight = get_matrix_weights(device)
        self.V.weight = get_matrix_weights(device)
        self.W.weight = get_matrix_weights(device)

        self.in_non_linearity = get_non_linearity().to(device)
        # self.query_non_linearity = get_non_linearity().to(device)
        self.query_non_linearity = self.in_non_linearity

        # Decoder
        # self.R = nn.Linear(vocab_size, embedding_dim, bias=False).to(device)
        self.R = nn.Linear(embedding_dim, vocab_size, bias=False).to(device)
        self.H = nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)
        # self.R.weight = nn.Parameter(get_r_matrix_weights(vocab_size, device), requires_grad=True).to(device)
        # self.H.weight = nn.Parameter(get_matrix_weights(device), requires_grad=True).to(device)
        self.R.weight = get_r_matrix_weights(vocab_size, device)
        self.H.weight = get_matrix_weights(device)

    def init_new_memories(self, keys, device, batch_size):
        # self.memories = torch.tensor(keys, requires_grad=True, device=device).repeat(batch_size, 1, 1)
        self.memories = keys.clone().detach().to(device).repeat(batch_size, 1, 1)

    def forward(self, batch):

        batch = self.embedding_matrix(batch)
        if (not learn_keys) and tie_keys:
            self.embedded_keys = nn.Parameter(self.embedding_matrix(self.keys))

        # re-initialize memories to key-values
        self.init_new_memories(self.embedded_keys, self.device, len(batch))

        # Encoder
        batch = batch * self.input_encoder_multiplier
        batch = batch.sum(dim=2)

        # Memory
        for sentence_idx in range(batch.shape[1]):
            sentence = batch[:, sentence_idx]
            sentence_memory_repeat = sentence.repeat(1, n_memories).view(len(batch), n_memories, -1)

            memory_gate = (sentence * self.memories.permute(1, 0, 2)).permute(1, 0, 2).sum(dim=2)
            key_gate = (sentence_memory_repeat * self.embedded_keys).sum(dim=2)
            gate = torch.sigmoid(memory_gate + key_gate)

            update_candidate = self.in_non_linearity(self.U(self.memories) + self.V(self.embedded_keys) + self.W(sentence_memory_repeat))
            # the null sentence mask make sure the padding sentences (that are not part of the original story, but are fake "null" sentences) doesn't effect the memories of th network
            null_sentence_mask = gate.clone().detach()
            null_sentence_mask[null_sentence_mask == 0.5] = 0
            null_sentence_mask[null_sentence_mask != 0] = 1
            # self.memories = self.memories + (update_candidate.permute(2, 0, 1) * gate).permute(1, 2, 0)
            self.memories = self.memories + (update_candidate.permute(2, 0, 1) * gate * null_sentence_mask).permute(1, 2, 0)
            self.memories = (self.memories.permute(2, 0, 1) / torch.norm(self.memories, dim=2)).permute(1, 2, 0)

    def decode(self, batch):

        batch = self.embedding_matrix(batch)
        # Decoder
        # query = query.to(device)
        batch = batch * self.query_encoder_multiplier
        batch = batch.sum(dim=1)
        answers_probabilities = F.softmax((batch * self.memories.permute(1, 0, 2)).sum(dim=2).t(), dim=0)
        scores = (self.memories.permute(2, 0, 1) * answers_probabilities).permute(1, 2, 0).sum(dim=1)
        results = self.R(self.query_non_linearity(batch + self.H(scores)))
        return results


def train(task, device):
    train, test = load_task(data_dir, task)

    vocab, vocab_size, entities = get_vocab_and_entities(task)

    global n_memories
    if not n_memories:
        n_memories = len(entities)

    print_start_train_message(task)

    models = [None] * n_tries
    optims = [None] * n_tries
    model_scores = [np.inf] * n_tries
    model_test_scores = [np.inf] * n_tries
    model_correct_scores = [0] * n_tries
    model_test_correct_scores = [0] * n_tries

    for try_idx in range(n_tries):
        embeddings_matrix, token_to_idx = init_embedding_matrix(vocab, device)
        keys = get_key_tensors(entities, embeddings_matrix, token_to_idx, device, tie_keys)

        len_max_sentence = get_len_max_sentence(train + test)
        # vec_train = vectorize_data(train, token_to_idx, len_max_sentence, len_max_story)
        vec_train = indexize_data(train, token_to_idx, len_max_sentence)
        vec_test = indexize_data(test, token_to_idx, len_max_sentence)

        entnet = EntNet(vocab_size, keys, len_max_sentence, embeddings_matrix, device)
        entnet.to(device)
        entnet = entnet.float()
        # entnet.load_state_dict(torch.load(STATE_PATH.format(task, 0)))

        ##### Define Loss and Optimizer #####
        criterion = nn.CrossEntropyLoss().to(device)
        learning_rate = 0.01
        optimizer = optim.Adam(entnet.parameters(), lr=learning_rate)
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(entnet.parameters(), lr=learning_rate)
        # optimizer.load_state_dict(torch.load(OPTIM_PATH.format(task, 0)))

        ##### Train Model #####
        epoch = 0
        permute_data = 'full'
        if teach:
            permute_data = 'no'
        loss_history = [np.inf] * max_stuck_epochs
        test_loss_history = [np.inf] * max_stuck_epochs
        correct_history = [0] * max_stuck_epochs
        test_correct_history = [0] * max_stuck_epochs
        net_history = [None] * max_stuck_epochs
        optim_history = [None] * max_stuck_epochs

        while True:
            epoch_loss = 0.0
            running_loss = 0.0
            correct_batch = 0
            correct_epoch = 0
            start_time = time.time()
            if teach and (epoch > 1 or loss_history[-1] < 0.3):
                permute_data = 'full'
            for i, batch in enumerate(batch_generator(vec_train, len_max_sentence, batch_size, permute_data)):
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

                nn.utils.clip_grad_value_(entnet.parameters(), gradient_clip_value)
                optimizer.step()
                # zero the parameter gradients
                optimizer.zero_grad()

                pred_idx = np.argmax(output.cpu().detach().numpy(), axis=1)
                for j in range(len(output)):
                    if pred_idx[j] == batch_answers[j].item():
                        correct_batch += 1
                        correct_epoch += 1

                if verbose:
                    if i % 50 == 49:
                        # print statistics
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                        running_loss = 0.0

                        print('[%d, %5d] correct: %d out of %d' % (epoch + 1, i + 1, correct_batch, 50 * batch_size))
                        correct_batch = 0

            # very loose approximation for the average loss over the epoch
            epoch_loss = epoch_loss / (len(vec_train) / batch_size)
            # print epoch time
            end_time = time.time()
            if verbose:
                print("###################################################################################################")
                print(end_time - start_time)
                print('epoch loss: %.3f' % epoch_loss)
                print("###################################################################################################")

            test_loss, test_correct = eval(task, device, entnet, vec_test, len_max_sentence)
            test_fail_rate = 100 - (float(test_correct)/len(vec_test)) * 100

            net_history.append(entnet.state_dict())
            optim_history.append(optimizer.state_dict())
            loss_history.append(epoch_loss)
            test_loss_history.append(test_loss)
            correct_history.append(correct_epoch)
            test_correct_history.append(test_correct)

            net_history = net_history[1:]
            optim_history = optim_history[1:]
            loss_history = loss_history[1:]
            test_loss_history = test_loss_history[1:]
            correct_history = correct_history[1:]
            test_correct_history = test_correct_history[1:]

            if (test_loss_history[0] - min(test_loss_history[1:]) < min_improvement) or test_fail_rate <= entnet_threshold[task - 1]:
                best_idx = np.argmin(test_loss_history)
                models[try_idx] = net_history[best_idx]
                optims[try_idx] = optim_history[best_idx]
                model_scores[try_idx] = loss_history[best_idx]
                model_test_scores[try_idx] = test_loss_history[best_idx]
                model_correct_scores[try_idx] = correct_history[best_idx]
                model_test_correct_scores[try_idx] = test_correct_history[best_idx]
                break

            # adjust learning rate every 25 epochs until 200 epochs
            if epoch < 200 and epoch % 25 == 24:
                learning_rate = learning_rate / 2
                optimizer = optim.Adam(entnet.parameters(), lr=learning_rate)
            if epoch == 200:
                best_idx = np.argmin(test_loss_history)
                models[try_idx] = net_history[best_idx]
                optims[try_idx] = optim_history[best_idx]
                model_scores[try_idx] = loss_history[best_idx]
                model_test_scores[try_idx] = test_loss_history[best_idx]
                model_correct_scores[try_idx] = correct_history[best_idx]
                model_test_correct_scores[try_idx] = test_correct_history[best_idx]
                break

            epoch += 1

        model_fail_rate = 100 - (float(model_test_correct_scores[try_idx]) / len(vec_test)) * 100
        if model_fail_rate <= entnet_threshold[task - 1]:
            break

    best_idx = np.argmin(model_test_scores)
    torch.save(models[best_idx], STATE_PATH.format(task))
    torch.save(optims[best_idx], OPTIM_PATH.format(task))

    print("Finished Training task {}\n".format(task) +
          "try {} was best\n".format(best_idx) +
          "loss is: {}\n".format(model_scores[best_idx]) +
          "correct: {} out of {}\n".format(model_correct_scores[best_idx], len(vec_train)) +
          "test loss is: {}\n".format(model_test_scores[best_idx]) +
          "test correct: {} out of {}\n".format(model_test_correct_scores[best_idx], len(vec_test)))


def eval(task, device, entnet, vec_test, len_max_sentence):
    ##### Define Loss and Optimizer #####
    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        running_loss = 0
        correct = 0
        start_time = time.time()
        for i, batch in enumerate(batch_generator(vec_test, len_max_sentence, batch_size, permute='no')):
            batch_stories, batch_queries, batch_answers = batch

            batch_stories, batch_queries, batch_answers = batch_stories.clone().detach().to(device), \
                                                          batch_queries.clone().detach().to(device), \
                                                          batch_answers.clone().detach().to(device)

            entnet(batch_stories)
            output = entnet.decode(batch_queries)
            loss = criterion(output, batch_answers)

            running_loss += loss.item()

            pred_idx = np.argmax(output.cpu().detach().numpy(), axis=1)
            for j in range(len(output)):
                if pred_idx[j] == batch_answers[j].item():
                    correct += 1

        # print epoch time
        # end_time = time.time()
        # print("###################################################################################################")
        # print(end_time - start_time)
        # print("###################################################################################################")

        # very loose approximation for the average loss over the epoch
        if verbose:
            print("Finished Testing task {}\n".format(task) +
                  "loss is: {}\n".format(running_loss / (len(vec_test) / batch_size)) +
                  "correct: {} out of {}\n".format(correct, len(vec_test)))

        return running_loss / (len(vec_test) / batch_size), correct

def test(task, device):
    train, test = load_task(data_dir, task)
    vocab, vocab_size, entities = get_vocab_and_entities(task)

    global n_memories
    if not n_memories:
        n_memories = len(entities)

    print_start_test_message(task)


    embeddings_matrix, token_to_idx = init_embedding_matrix(vocab, device)
    keys = get_key_tensors(entities, embeddings_matrix, token_to_idx, device, tie_keys)

    len_max_sentence = get_len_max_sentence(train + test)
    vec_test = indexize_data(test, token_to_idx, len_max_sentence)

    entnet = EntNet(vocab_size, keys, len_max_sentence, embeddings_matrix, device)
    entnet.to(device)
    entnet = entnet.float()
    entnet.load_state_dict(torch.load(STATE_PATH.format(task, 0)))

    loss, correct = eval(task, device, entnet, vec_test, len_max_sentence)

    if not verbose:
        print("Finished Testing task {}\n".format(task) +
              "loss is: {}\n".format(loss) +
              "correct: {} out of {}\n".format(correct, len(vec_test)))

def main():

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument(
        "--verbose",
        help="increases the verbosity of the output",
        action="store_true"
    )

    parser.add_argument(
        '--tasks',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        help='the tasks to learn')

    parser.add_argument(
        "--embedding_dim",
        help="the dimension of the emmbeding space for the vocabulary",
        type=int,
        default=100
    )

    parser.add_argument(
        "--n_memories",
        help="the number of memories to keep in the net",
        type=int,
    )

    parser.add_argument(
        "--batch_size",
        help="the size of the mini-batches in the learning proccess",
        type=int,
        default=32
    )

    parser.add_argument(
        "--gradient_clip_value",
        help="the value at which to clip the gradients",
        type=int,
        default=40
    )

    parser.add_argument(
        "--max_stuck_epochs",
        help="the maximal number of consecutive epochs the training is allowed to have no improvement, before it stops",
        type=int,
        default=6
    )

    parser.add_argument(
        "--teach",
        help="increases the verbosity of the output",
        action="store_true"
    )

    parser.add_argument(
        "--min_improvement",
        help="the minimal improvement in score between 2 epoch so that they are'nt considered stuck",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--optimizer_name",
        help="the name of the optimizer to use",
        type=str,
        default="adam"
    )

    parser.add_argument(
        "--random_seed",
        help="the value of the random seed, if it should be set",
        type=int
    )

    parser.add_argument(
        "--no_tie_keys",
        help="sets the initial key values in the EntNet to random values, instead of tying them to the words in the vocabulary",
        action="store_true"
    )

    # TODO: this option doesn't really work unless specifiyng "retain_graph=True" when calling "backward". fix this.
    parser.add_argument(
        "--no_learn_keys",
        help="disables the keys in the EntNet being learned as parameters",
        action="store_true"
    )

    parser.add_argument(
        "--state_path",
        help="the path to load from or save to the EntNet",
        type=str,
        default="./trained_models/task_{}.pth"
    )

    parser.add_argument(
        "--optim_path",
        help="the path to load from or save to the optimizer of the EntNet",
        type=str,
        default="./trained_models/optim_{}.pth"
    )

    parser.add_argument(
        "--cpu",
        help="trains the net on CPU",
        action="store_true"
    )

    parser.add_argument(
        "--train",
        help="trains the net",
        action="store_true"
    )

    parser.add_argument(
        "--test",
        help="tests the net",
        action="store_true"
    )

    parser.add_argument(
        "--load_net",
        help="used saved trained nets to test or complete training",
        action="store_true"
    )

    parser.add_argument(
        "--n_tries",
        help="the number of time the net is trained on each task. the try with the maximal results is chosen and saved",
        type=int,
        default=10
    )

    curr_dir = os.getcwd()
    args = parser.parse_args()

    global verbose, embedding_dim, n_memories, batch_size, gradient_clip_value, max_stuck_epochs, teach,\
        min_improvement, optimizer_name, tie_keys, learn_keys, STATE_PATH, OPTIM_PATH, random_seed, n_tries, cuda

    verbose = args.verbose
    embedding_dim = args.embedding_dim
    n_memories = args.n_memories
    batch_size = args.batch_size
    gradient_clip_value = args.gradient_clip_value
    max_stuck_epochs = args.max_stuck_epochs
    teach = args.teach
    min_improvement = args.min_improvement
    optimizer_name = args.optimizer_name
    tie_keys = not args.no_tie_keys
    learn_keys = not args.no_learn_keys
    tasks = args.tasks
    STATE_PATH = args.state_path
    OPTIM_PATH = args.optim_path
    n_tries = args.n_tries

    # for reproducibility
    if args.random_seed:
        random_seed = args.random_seed
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        seed(args.random_seed)
    else:
        random_seed = "not set"

    device = None
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        cuda = False

    for task in tasks:
        if args.train:
            train(task, device)
        if args.test:
            test(task, device)


if __name__ == "__main__":
    main()