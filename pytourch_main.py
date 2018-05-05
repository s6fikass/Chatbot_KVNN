import unicodedata
import string
import re
import random
import time
import datetime
import math
import socket

from corpus.textdata import TextData
from model.seq2seq_model import EncoderRNN, LuongAttnDecoderRNN

hostname = socket.gethostname()

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

USE_CUDA = False

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    length = Variable(torch.LongTensor(length)).cuda()

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


PAD_token = 0
EOS_token = 1
SOS_token = 2
EOU_token = 3


hidden_size = 256

train_file = 'data/kvret_train_public.json'
valid_file = 'data/kvret_dev_public.json'
test_file = 'data/kvret_test_public.json'
model_dir = "pytourch_trained_model"

textData = TextData(train_file, valid_file, test_file)
batch_size = 250
batches = textData.getBatches(batch_size, transpose=False)
for current_step in range(0, len(batches)):
    nextBatch = batches[current_step]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(nextBatch.encoderSeqs)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(nextBatch.decoderSeqs)).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    small_hidden_size = 8
    small_n_layers = 2

    encoder_test = EncoderRNN(textData.getVocabularySize(), small_hidden_size, small_n_layers)
    decoder_test = LuongAttnDecoderRNN('general', small_hidden_size, textData.getVocabularySize(), small_n_layers)

    if USE_CUDA:
        encoder_test.cuda()
        decoder_test.cuda()
    input_lengths = [textData.getInputMaxLength() for s in nextBatch.encoderSeqs]
    encoder_outputs, encoder_hidden = encoder_test(input_var, input_lengths, None)

    print('encoder_outputs', encoder_outputs.size())  # max_len x batch_size x hidden_size
    print('encoder_hidden', encoder_hidden.size())  # n_layers * 2 x batch_size x hidden_size

    max_target_length = textData.getTargetMaxLength()

    # Prepare decoder input and outputs
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder_test.n_layers]  # Use last (forward) hidden state from encoder
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder_test.output_size))

    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()
        decoder_input = decoder_input.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder_test(
            decoder_input, decoder_hidden, encoder_outputs
        )
        all_decoder_outputs[t] = decoder_output  # Store this step's outputs
        decoder_input = input_var[t]  # Next input is current target
    target_lengths = [textData.getTargetMaxLength() for s in nextBatch.encoderSeqs]

    # Test masked cross entropy loss
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),
        input_var.transpose(0, 1).contiguous(),
        target_lengths
    )
    print('loss', loss.data[0])

    break




