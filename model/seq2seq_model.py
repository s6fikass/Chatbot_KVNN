import unicodedata
import string
import re
import random
import time
import datetime
import math
import socket
hostname = socket.gethostname()

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
#from masked_cross_entropy import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=self.dropout)

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.lstm(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)  # unpack (back to padded)
        return output, hidden


class KVEncoderRNN(nn.Module):
    def __init__(self, input1_size,input2_size, hidden_size, n_layers=1, dropout=0.1):
        super(KVEncoderRNN, self).__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding1 = nn.Embedding(input1_size, hidden_size)
        self.embedding2 = nn.Embedding(input2_size, hidden_size)
        self.lstm = nn.LSTM(input1_size, hidden_size)

    def forward(self, input_seqs, input_kb_seqs, hidden=None):

        embedded = self.embedding1(input_seqs)
        kb_embedded = self.embedding2(input_kb_seqs)
        output, hidden = self.lstm(embedded, hidden)

        return output, hidden, kb_embedded

class Attn(nn.Module):
    def __init__(self, method, hidden_size, use_cuda=None):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.cuda = use_cuda

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)

        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))  # B x S

        if self.cuda:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=0).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':

            energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
            return energy

        elif self.method == 'general':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.dot(self.v.view(-1), energy.view(-1))
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy

class KbAttn(nn.Module):
    def __init__(self, hidden_size):
        super(KbAttn, self).__init__()

        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, k_embedding):

        max_len = 431

        this_batch_size = hidden.size(1)

        print("max_len",max_len)
        print("this_batch_size", this_batch_size)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))  # B x S

        # if USE_CUDA:
        #     attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):

                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return attn_energies

    def score(self, hidden, encoder_output):
        print(torch.cat((hidden, encoder_output), 1).shape)
        energy = self.attn(torch.cat((hidden, encoder_output), 1))
        energy = self.v.dot(energy)
        return energy


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1, intent_size=3, use_cuda=None):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.intent_size = intent_size
        self.use_cuda=use_cuda

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.lstm_intent = nn.LSTM(intent_size, hidden_size)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.intent_out = nn.Linear(self.hidden_size * 2, self.intent_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size,use_cuda)
            self.intent_attn = Attn(attn_model, hidden_size, use_cuda)

    def forward(self, input_seq, last_context, last_hidden, encoder_outputs, intent_batch=False, Kb_batch=False):
        # Note: we run this one step at a time (in order to do teacher forcing)

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
#         print('[decoder] input_seq', input_seq.size()) # batch_size x 1
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N
#         print('[decoder] word_embedded', embedded.size())

        # Get current hidden state from input word and last hidden state
#         print('[decoder] last_hidden', last_hidden.size())

        rnn_output, hidden = self.lstm(embedded, last_hidden)
        # print('[decoder] rnn_output', rnn_output.size())

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
#        print('[decoder] attn_weights', attn_weights.size())
#         print('[decoder] encoder_outputs', encoder_outputs.size())
        intent_score = None
        if intent_batch:
            # new_hidden = Variable(torch.zeros(self.n_layers*1, batch_size, self.hidden_size))
            # if self.use_cuda:
            #     intent_hidden = new_hidden.cuda()
            # else:
            #     intent_hidden = new_hidden
            intent_hidden = hidden[0].clone()
            # print("intent_intent_hidden", intent_hidden.shape)
            intent_attn_weights = self.intent_attn(intent_hidden, encoder_outputs)
            intent_context = intent_attn_weights.bmm(encoder_outputs.transpose(0, 1))
            concated = torch.cat((intent_hidden, intent_context.transpose(0, 1)), 2)  # 1,B,D
            intent_score = self.intent_out(concated.squeeze(0))  # B,D

        if Kb_batch:
            # new_hidden = Variable(torch.zeros(self.n_layers*1, batch_size, self.hidden_size))
            # if self.use_cuda:
            #     intent_hidden = new_hidden.cuda()
            # else:
            #     intent_hidden = new_hidden
            kb_hidden = hidden[0].clone()
            # print("intent_intent_hidden", intent_hidden.shape)
            intent_attn_weights = self.intent_attn(kb_hidden, encoder_outputs)
            intent_context = intent_attn_weights.bmm(encoder_outputs.transpose(0, 1))
            concated = torch.cat((intent_hidden, intent_context.transpose(0, 1)), 2)  # 1,B,D
            intent_score = self.intent_out(concated.squeeze(0))  # B,D


        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N
#         print('[decoder] context', context.size())


        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
#         print('[decoder] rnn_output', rnn_output.size())
#         print('[decoder] context', context.size())

        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6)
#         output = F.log_softmax(self.out(concat_output))
        output = self.out(concat_output)


        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights, intent_score

class KVAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(KVAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_kb = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.lstm = nn.LSTM(hidden_size,hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
            self.kbattn = KbAttn(hidden_size)

    def reshapeKb(self,kb_embeding):
        embedding = torch.sum(kb_embeding,dim=2)
        return embedding.reshape(10, self.hidden_size, 431)

    def kbLogits(self, kb, batch_size, pad_length):

        # Create variable to store attention energies of 0 for non kb entities
        v = Variable(torch.zeros(batch_size, pad_length, 1523))  # B x S x Vocab_size
        print(kb.shape)
        attn_energies = kb.reshape(batch_size, pad_length, 431)
        tensor = torch.cat([v, attn_energies], axis=2)
        print(tensor.shape)
        return tensor

    def forward(self, input_seq, kb_inputs, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time (in order to do teacher forcing)

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        print('[decoder] input_seq', input_seq.size()) # batch_size x 1

        # Decoder Embedding
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N
        print('[decoder] word_embedded', embedded.size())

        # Get current hidden state from input word and last hidden state

        print('[decoder] last_hidden', last_hidden.size())
        rnn_output, hidden = self.gru(embedded, last_hidden)
        print('[decoder] rnn_output', rnn_output.size())

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        print('[decoder] attn_weights', attn_weights.size())
        print('[decoder] encoder_outputs', encoder_outputs.size())


        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

        print('[decoder] context', context.size())

        embedded2 = self.embedding_kb(kb_inputs)
        print('[KB] word_embedded', embedded2.size())
        embedded2 = self.reshapeKb(embedded2)
        print('[KB] reshaped_word_embedded', embedded2.size())

        print ("calculating W1 [ kj, ~hi] " )
        kb_attn = self.kbattn(embedded2, last_hidden)

        print(kb_attn)



        kb_attn = self.kbLogits(embedded2,batch_size, encoder_outputs.size(0))

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
#         print('[decoder] rnn_output', rnn_output.size())
#         print('[decoder] context', context.size())
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6)
#         output = F.log_softmax(self.out(concat_output))
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights, kb_attn