#imports
import pickle
import numpy as np
from collections import defaultdict
import os
from collections import OrderedDict
import torch
#from args import get_args
import json
import itertools
import re
import pandas as pd
import argparse
#args = get_args()
import spacy
nlp=spacy.load('en_core_web_sm')

class DialogBatcher:
    """
    Wrapper for batching the Soccer Dialogue dataset
    """
    def __init__(self, gpu=True, max_sent_len=30, max_resp_len=20, batch_size=32):
        self.batch_size = batch_size
        #self.use_mask = use_mask
        self.gpu = gpu
        self.max_sent_len = max_sent_len
        self.max_resp_len = max_resp_len

        #self.vocab_glove = np.load(args.vocab_glove).item()
        vec_dim = 300

        #self.stoi['EOS'] = len(self.stoi)+1
        #self.stoi['SOS'] = len(self.stoi)+1

        #get required dictionaries for data
        tr, self.stoi = self.read_dat('data/samples/train.csv', train=True)
        te = self.read_dat('data/samples/test.csv')
        val = self.read_dat('data/samples/valid.csv')
        # self.all = self.get_sequences('all')
        self.stoi['<pad>'] = 0
        self.stoi['<unknown>'] = len(self.stoi)+1
        self.stoi['<go>'] = len(self.stoi)+1
        self.stoi['<eos>'] = len(self.stoi)+1
        self.train = self.get_sequences(tr)
        self.test = self.get_sequences(te)
        self.valid = self.get_sequences(val)

        self.n_words = len(self.stoi) + 1
        self.n_train = len(self.train['x'])
        self.n_val = len(self.valid['x'])
        self.n_test = len(self.test['x'])
        # self.n_all = len(self.all['x'])

        self.itos = {v: k for k, v in self.stoi.items()}
        self.vocab_glove = defaultdict(list)
        with open('data/samples/jointEmbedding.txt', 'r') as f:
            joint_emb = f.readlines()
        for l in joint_emb:
            l = l.replace('\n', '').split()
            word = l[0]
            vec = l[1:]
            self.vocab_glove[word] = vec

        # get pretrained vectors
        self.vectors = np.zeros((len(self.itos)+1, vec_dim))
        for k, v in self.vocab_glove.items():
            #self.vectors[self.stoi[k.encode('utf-8')]] = v
            try:
                self.vectors[self.stoi[k]] = v
            except KeyError:
                continue

        self.vectors = torch.from_numpy(self.vectors.astype(np.float32))

    def read_dat(self, filename, train=False):
        ds = {}
        ds['x'], ds['y'] = [], []
        dat = pd.read_csv(filename, header=None)
        ds['x'] = np.array(dat[0])
        ds['y'] = np.array(dat[1])

        if train:
            # Create vocabulary
            vocab = defaultdict(float)
            for s in ds['x']:
                #print (s)
                for w in s.split():
                    vocab[w] += 1.0
            for s in ds['y']:
                if isinstance(s, str):
                #print (s)
                    for w in s.split():
                        vocab[w] += 1.0
            stoi = dict(zip(vocab.keys(), range(1, len(vocab)+1)))
            return ds, stoi
        else:
            return ds

    def get_sequences(self, dat):
        ds = {}
        ds['x'] = [[self.getw2i(w) for w in s.split()] for s in dat['x']]
        ds['y'] = [[self.getw2i(w) for w in s.split()] for s in dat['y'] if isinstance(s, str)]
        #print (ds['y'])
        #print (ds['x'][0])
        ds['x'] = [(idxs + [self.stoi['<eos>']]) for idxs in ds['x']]
        #print (ds['x'][0])
        ds['y'] = [(idxs + [self.stoi['<eos>']]) for idxs in ds['y']]
        #print (ds['y'])
        return ds

    def tokenize(self, sentence):
        if isinstance(sentence, str):
            return (self.getw2i(w) for w in sentence.split())
        else:
            return []

    def getw2i(self, word):
        try:
            return self.stoi[word]
        except KeyError:
            return self.stoi['<unknown>']

    def geti2w(self, word):
        """
        get id 2 word
        :param word:
        :return:
        """
        if self.gpu:
            word = self.itos[int(word.cpu().numpy())]
            if isinstance(word, str):
                return word
            else:
                return word
        else:
            #word = self.itos[int(word.numpy())].decode('utf-8')
            word = self.itos[int(word.numpy())]
            if isinstance(word, str):
                return word
            else:
                return word

    def get_iter(self, dataset='train'):
        # get iterations.
        if dataset == 'train':
            dataset = self.train
        elif dataset == 'valid':
            dataset = self.valid
        else:
            dataset = self.test

        for i in range(0, len(dataset['x']), self.batch_size):
            query = dataset['x'][i:i+self.batch_size]
            response = dataset['y'][i:i+self.batch_size]
            #team = dataset['team'][i:i + self.batch_size]

            x, y, mx, my = self._load_batch(query, response, self.batch_size)

            yield x, y, mx, my

    def _load_batch(self, q, a, b_s):
        b_s = min(b_s, len(q))
        max_len_q = np.max([len(sent) for sent in q])
        max_len_q = (max_len_q) if max_len_q < self.max_sent_len else self.max_sent_len
        max_len_a = np.max([len(sent) for sent in a])
        max_len_a = (max_len_a) if max_len_a < self.max_resp_len else self.max_resp_len
        x = np.zeros([max_len_q, b_s], np.int)
        y = np.zeros([max_len_a, b_s], np.int)

        x_mask = np.zeros([max_len_q, b_s], np.int)
        y_mask = np.zeros([max_len_a, b_s], np.int)

        for j, (row_t, row_l) in enumerate(zip(q, a)):
            row_t = row_t[-max_len_q:]
            row_l = row_l[:max_len_a]
            #print (row_t, len(row_t))
            x[:len(row_t), j] = row_t
            y[:len(row_l), j] = row_l
            x_mask[:len(row_t), j] = 1
            y_mask[:len(row_l), j] = 1


        x_o = torch.from_numpy(x)
        y_o = torch.from_numpy(y).type(torch.FloatTensor)
        x_mask = torch.from_numpy(x_mask).type(torch.FloatTensor)
        y_mask = torch.from_numpy(y_mask).type(torch.FloatTensor)

        if self.gpu:
            x_o, y_o, x_mask, y_mask = x_o.cuda(), y_o.cuda(), x_mask.cuda(), y_mask.cuda()

        return x_o, y_o, x_mask, y_mask


if __name__ == '__main__':
    batcher = DialogBatcher(gpu=False)
    batches = batcher.get_iter('valid')
    for b in batches:
        print (b)

