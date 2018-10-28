# coding: utf-8
import csv
import logging
import re
from collections import Counter
import  os
import pickle
import numpy as np
import torch
from pandas import read_csv
from sklearn.datasets import fetch_20newsgroups
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm
import nltk

# Hyperparameters
N_EMBEDDING = 300
BASE_STD = 0.01
BATCH_SIZE = 512
NUM_EPOCH = 10
MIN_WORD_OCCURENCES = 1
X_MAX = 100
ALPHA = 0.75
BETA = 0.0001
RIGHT_WINDOW = 20

USE_CUDA = False

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


def cuda(x):
    if USE_CUDA:
        return x.cuda()
    return x


class WordIndexer:
    """Transform g a dataset of text to a list of index of words. Not memory 
    optimized for big datasets"""

    def __init__(self, min_word_occurences=1, right_window=10):
        self.right_window = right_window
        self.min_word_occurences = min_word_occurences
        self.word_occurrences = {}
        self.loadDataset("data/samples/dataset-kvret.pkl")
        #self.re_words = nltk.word_tokenize() #re.compile(r"\b[a-zA-Z1-9]{1,}\b")

    def loadDataset(self, filename):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word_to_index = data['word2id']
            self.index_to_word = data['id2word']
            self.intent2id=data['intent2id']
            self.id2intent=data['id2intent']
            self.idCount = data.get('idCount', None)
            self.trainingSamples = data['trainingSamples']
            self.validationSamples = data['validationSamples']
            self.testSamples = data['testSamples']
            self.unknownToken = self.word_to_index['<unknown>']  # Restore special words

    def _get_or_set_word_to_index(self, word):
        try:
            return self.word_to_index[word]
        except KeyError:
            idx = len(self.word_to_index)
            self.word_to_index[word] = idx
            self.index_to_word[idx] = word
            return idx

    @property
    def n_words(self):
        return len(self.word_to_index)

    def fit_transform(self, texts):
        l_words = [list(re.findall(r"[\w']+|[^\s\w']", ' '.join(re.split('_', sentence.lower())).replace(',', ' ').strip()))
                   for sentence in texts]
        word_occurrences = Counter(word for words in l_words for word in words)

        self.word_occurrences = {
            word: n_occurences
            for word, n_occurences in word_occurrences.items()
            if n_occurences >= self.min_word_occurences}

        return [[self._get_or_set_word_to_index(word)
                 if word in self.word_occurrences else self.unknownToken
                 for word in words]
                for words in l_words]

    def _get_ngrams(self, indexes):
        for i, left_index in enumerate(indexes):
            window = indexes[i + 1:i + self.right_window + 1]
            for distance, right_index in enumerate(window):
                yield left_index, right_index, distance + 1

    def get_comatrix(self, data):
        comatrix = Counter()
        z = 0
        for indexes in data:
            l_ngrams = self._get_ngrams(indexes)
            for left_index, right_index, distance in l_ngrams:
                comatrix[(left_index, right_index)] += 1. / distance
                z += 1
        return zip(*[(left, right, x) for (left, right), x in comatrix.items()])


class GloveDataset(Dataset):
    def __len__(self):
        return self.n_obs

    def __getitem__(self, index):

        return (self.L_vecs[index].data + self.R_vecs[index].data).numpy()

    def __init__(self, texts, right_window=1, random_state=0):
        torch.manual_seed(random_state)

        self.indexer = WordIndexer(right_window=right_window,
                                   min_word_occurences=MIN_WORD_OCCURENCES)
        data = self.indexer.fit_transform(texts)
        left, right, n_occurrences = self.indexer.get_comatrix(data)
        n_occurrences = np.array(n_occurrences)
        self.n_obs = len(left)

        # We create the variables
        self.L_words = cuda(torch.LongTensor(left))
        self.R_words = cuda(torch.LongTensor(right))

        self.weights = np.minimum((n_occurrences / X_MAX) ** ALPHA, 1)
        self.weights = Variable(cuda(torch.FloatTensor(self.weights)))
        self.y = Variable(cuda(torch.FloatTensor(np.log(n_occurrences))))

        # We create the embeddings and biases
        N_WORDS = self.indexer.n_words
        L_vecs = cuda(torch.randn((N_WORDS, N_EMBEDDING)) * BASE_STD)
        R_vecs = cuda(torch.randn((N_WORDS, N_EMBEDDING)) * BASE_STD)
        L_biases = cuda(torch.randn((N_WORDS,)) * BASE_STD)
        R_biases = cuda(torch.randn((N_WORDS,)) * BASE_STD)
        self.all_params = [Variable(e, requires_grad=True)
                           for e in (L_vecs, R_vecs, L_biases, R_biases)]
        self.L_vecs, self.R_vecs, self.L_biases, self.R_biases = self.all_params


def gen_batchs(data):
    """Batch sampling function"""
    indices = torch.randperm(len(data))
    if USE_CUDA:
        indices = indices.cuda()
    for idx in range(0, len(data) - BATCH_SIZE + 1, BATCH_SIZE):
        sample = indices[idx:idx + BATCH_SIZE]
        l_words, r_words = data.L_words[sample], data.R_words[sample]
        l_vecs = data.L_vecs[l_words]
        r_vecs = data.R_vecs[r_words]
        l_bias = data.L_biases[l_words]
        r_bias = data.R_biases[r_words]
        weight = data.weights[sample]
        y = data.y[sample]
        yield weight, l_vecs, r_vecs, y, l_bias, r_bias


def get_loss(weight, l_vecs, r_vecs, log_covals, l_bias, r_bias):
    sim = (l_vecs * r_vecs).sum(1).view(-1)
    x = (sim + l_bias + r_bias - log_covals) ** 2
    loss = torch.mul(x, weight)
    return loss.mean()


def train_model(data: GloveDataset):
    optimizer = torch.optim.Adam(data.all_params, weight_decay=1e-8)
    optimizer.zero_grad()
    for epoch in tqdm(range(NUM_EPOCH)):
        logging.info("Start epoch %i", epoch)
        num_batches = int(len(data) / BATCH_SIZE)
        avg_loss = 0.0
        n_batch = int(len(data) / BATCH_SIZE)
        for batch in tqdm(gen_batchs(data), total=n_batch, mininterval=1):
            optimizer.zero_grad()
            loss = get_loss(*batch)
            avg_loss += loss.item() / num_batches
            loss.backward()
            optimizer.step()
        logging.info("Average loss for epoch %i: %.5f", epoch + 1, avg_loss)


if __name__ == "__main__":
    logging.info("Fetching data")
    #newsgroup = fetch_20newsgroups(data_home="data/glove_data",remove=('headers', 'footers', 'quotes'))
    logging.info("Build dataset")
    with open('data/samples/train.csv', 'r') as myfile:
        data=myfile.readlines()
    glove_data = GloveDataset(data, right_window=RIGHT_WINDOW)
    logging.info("#Words: %s", glove_data.indexer.n_words)
    logging.info("#Ngrams: %s", len(glove_data))
    logging.info("Start training")
    train_model(glove_data)
    print(glove_data.__getitem__(1))
    for key, value in glove_data.indexer.index_to_word:
        print(key)
        print(value)
    #     myfile.write("%s\n" % var1)
    # glove_data.indexer.index_to_word[]

    # word_inds = np.random.choice(np.arange(len(vocab)), size=10, replace=False)
    # for word_ind in word_inds:
    #     # Create embedding by summing left and right embeddings
    #     w_embed = (l_embed[word_ind].data + r_embed[word_ind].data).numpy()
    #     x, y = w_embed[0][0], w_embed[1][0]
    #     plt.scatter(x, y)
    #     plt.annotate(vocab[word_ind], xy=(x, y), xytext=(5, 2),
    #                  textcoords='offset points', ha='right', va='bottom')