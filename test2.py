import unicodedata
import string
import re
import random
import time
import datetime
import math
import socket
import os
from json import JSONDecoder
from reader import Data,Vocabulary
import argparse
from model.seq2seq_model import KVEncoderRNN, KVAttnDecoderRNN, EncoderRNN, LuongAttnDecoderRNN

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
#from masked_cross_entropy import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
#matplotlib inline
import torch
from torch.nn import functional
from torch.autograd import Variable


vocab = Vocabulary('data/vocabulary.json',
                   padding=20)
kb_vocab = Vocabulary('data/vocabulary.json',
                      padding=4)

print('Loading datasets.')
training = Data('./data/train_data.csv', vocab, kb_vocab)
# validation = Data(args.validation_data, vocab, kb_vocab)
training.load()
# validation.load()
training.transform()
training.kb_out()
# validation.transform()
# validation.kb_out()
training_generator = training.random_batch(10)
hidden_size=200
n_layers = 2
dropout = 0.1
hidden_size = 200
batch_size = 10

# Configure training/optimization

learning_rate = 0.0001
decoder_learning_ratio = 5.0

x, kb, y = next(training_generator)

# Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
target_lengths = [len(s) for s in y]
input_batch = Variable(torch.LongTensor(x)).transpose(0, 1)
target_batch = Variable(torch.LongTensor(y)).transpose(0, 1)
kb_batch = Variable(torch.LongTensor(kb))


encoder = EncoderRNN(vocab.size(), hidden_size , n_layers, dropout=dropout)
decoder = KVAttnDecoderRNN('dot', hidden_size, vocab.size(), n_layers, dropout=dropout)

encoder_outputs, encoder_hidden = encoder(input_batch, None)
print (encoder_hidden.shape)
print (encoder_outputs.shape)

# Prepare input and output variables

decoder_input = Variable(torch.LongTensor([[0] * batch_size])).transpose(0, 1)
#     print('decoder_input', decoder_input.size())
decoder_context = encoder_outputs[-1]
decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

max_target_length = target_batch.shape[0]
all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))


if True:
    # Run through decoder one time step at a time
    for t in range(max_target_length):

        decoder_output, decoder_context, decoder_hidden, decoder_attn, kb_attn = decoder(
            decoder_input, kb_batch, decoder_context, decoder_hidden, encoder_outputs
        )


        all_decoder_outputs[t] = decoder_output

        decoder_input = target_batch[t]

