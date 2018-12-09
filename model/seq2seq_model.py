import unicodedata
import string
import re
import random
import time
import datetime
import math
import socket
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from util.utils import masked_cross_entropy, compute_ent_loss
import os

from util.measures import moses_multi_bleu
import nltk

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

hostname = socket.gethostname()


class LuongEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, emb_dim, b_size, n_layers=1, dropout=0.1, gpu=False):
        super(LuongEncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gpu = gpu
        self.n_layers = n_layers
        self.dropout = dropout
        self.emb_dim= emb_dim
        self.b_size = b_size
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=self.dropout)

    def init_weights(self, b_size):
        # intiialize hidden weights
        c0 = Variable(torch.zeros(self.n_layers, b_size, self.hidden_size))
        h0 = Variable(torch.zeros(self.n_layers, b_size, self.hidden_size))

        if self.gpu:
            c0 = c0.cuda()
            h0 = h0.cuda()

        return h0, c0

    def forward(self, embedded, input_lengths, hidden=None):
        hidden = self.init_weights(embedded.size(1))
        #packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.lstm(embedded, hidden)
        #output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)  # unpack (back to padded)
        return output, hidden

class EncoderRNN(nn.Module):
    """
    Encoder RNN module
    """

    def __init__(self, input_size, emb_size, hidden_size, b_size, vocab_size, n_layers=1, dropout=0.0, emb_drop=0.0,
                 gpu=False):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_cuda = gpu
        self.b_size = b_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.gpu = gpu

        self.embedding_dropout = nn.Dropout(emb_drop)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, dropout=self.dropout)
        # self.rnn = rnn

    def init_weights(self, b_size):
        # intiialize hidden weights
        c0 = Variable(torch.zeros(self.n_layers, b_size, self.hidden_size))
        h0 = Variable(torch.zeros(self.n_layers, b_size, self.hidden_size))

        if self.gpu:
            c0 = c0.cuda()
            h0 = h0.cuda()

        return h0, c0

    def forward(self, inp_emb, input_lengths=None):
        # input_q = S X B X EMB
        # embedded = self.embedding(input_q)
        embedded = self.embedding_dropout(inp_emb)
        hidden = self.init_weights(inp_emb.size(1))

        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        outputs, hidden = self.rnn(embedded, hidden)  # outputs = S X B X n_layers*H, hidden = 2 * [1 X B X H]
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        return outputs, hidden


# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size, gpu):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            if gpu:
                self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size).cuda())
            else:
                self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class Attention(nn.Module):
    """
    Attention mechanism (Luong)
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # weights
        self.W_h = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.epsilon = 1e-5

    def forward(self, encoder_outputs, decoder_hidden, inp_mask):
        seq_len = encoder_outputs.size(1)  # get sequence lengths S
        H = decoder_hidden.repeat(seq_len, 1, 1).transpose(0, 1)  # B X S X H
        energy = torch.tanh(self.W_h(torch.cat([H, encoder_outputs], 2)))  # B X S X H
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B X 1 X H]
        energy = torch.bmm(v, energy).view(-1, seq_len)  # [B X T]

        a = F.softmax(energy,dim=0) * inp_mask.transpose(0, 1)  # B X T
        normalization_factor = a.sum(1, keepdim=True)
        a = a / (normalization_factor + self.epsilon)  # adding a small offset to avoid nan values
        aaa= a/self.epsilon
        a = a.unsqueeze(1)
        context = a.bmm(encoder_outputs)

        return a, context


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, emb_dim, output_size, batch_size, n_layers=1, dropout=0.1, intent_size=3, emb=None,
                 use_cuda=None):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.intent_size = intent_size
        self.use_cuda = use_cuda
        self.emb_dim = emb_dim
        self.batch_size=batch_size
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.lstm_intent = nn.LSTM(intent_size, hidden_size)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.intent_out = nn.Linear(self.hidden_size , self.intent_size)

        # Choose attention model
        if attn_model != 'none':
            #self.attn = Attn(attn_model, hidden_size, use_cuda)
            self.intent_attn = Attn(attn_model, hidden_size, self.use_cuda) #Attn(attn_model, hidden_size, use_cuda)
            self.attention = Attention(hidden_size)

    def forward(self, embedded, last_context, last_hidden, encoder_outputs, inp_mask, intent_batch=False, Kb_batch=False):
        # Note: we run this one step at a time (in order to do teacher forcing)

        # Get the embedding of the current input word (last output word)
        batch_size = self.batch_size
        #         print('[decoder] input_seq', input_seq.size()) # batch_size x 1


        embedded = embedded.view(1, batch_size, self.hidden_size)  # S=1 x B x N
        #print('[decoder] word_embedded', embedded.size())

        # Get current hidden state from input word and last hidden state
        #         print('[decoder] last_hidden', last_hidden.size())

        rnn_output, hidden = self.lstm(embedded, last_hidden)
        # print('[decoder] rnn_output', rnn_output.size())

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        #attn_weights = self.attn(rnn_output, encoder_outputs)
        #        print('[decoder] attn_weights', attn_weights.size())
        #         print('[decoder] encoder_outputs', encoder_outputs.size())
        intent_score = None
        if intent_batch:
            # new_hidden = Variable(torch.zeros(self.n_layers*1, batch_size, self.hidden_size))
            # if self.use_cuda:
            #     intent_hidden = new_hidden.cuda()
            # else:
            #     intent_hidden = new_hidden
            intent_hidden = last_hidden[0]
            #print("intent_intent_hidden", intent_hidden.shape)

            # intent_attn_weights= self.intent_attn(intent_hidden, encoder_outputs)
            # intent_context = intent_attn_weights.bmm(encoder_outputs.transpose(0, 1))
            intent_attn_output = self.attention_net(encoder_outputs.transpose(0, 1), intent_hidden)
        #     print(intent_attn_output.transpose(0, 1).shape)
        #    concated = torch.cat((intent_hidden.squeeze(0), intent_attn_output.transpose(0, 1)), 2)  # 1,B,D

            #print("concated",intent_attn_output.shape)
            intent_score = self.intent_out(intent_attn_output)#last_hidden[-1].squeeze(0))#  # B,D

        s_t = hidden[0][-1].unsqueeze(0)
        alpha, context = self.attention(encoder_outputs.transpose(0,1), s_t, inp_mask)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, alpha, intent_score

    def attention_net(self, lstm_output, final_state):

        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

class Decoder(nn.Module):
    """
    Decoder RNN
    """
    def __init__(self, hidden_size, emb_dim, vocab_size, n_layers=1, dropout=0.0):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        #self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hidden_size, n_layers, dropout=dropout)
        #self.rnn = rnn
        self.out = nn.Linear(hidden_size, vocab_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)

        # Attention
        self.attention = Attention(hidden_size)

    def forward(self, inp_emb, last_hidden, encoder_outputs, inp_mask):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)

        batch_size = min(last_hidden[0].size(1), inp_emb.size(0))
        inp_emb = inp_emb[-batch_size:]

        #max_len = encoder_outputs.size(0)args
        encoder_outputs = encoder_outputs.transpose(0,1) # B X S X H
        #embedded = self.embedding(input_seq)
        embedded = self.dropout(inp_emb)
        embedded = embedded.view(1, batch_size, self.emb_dim) # S=1 x B x N
        #print (embedded.size())
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.rnn(embedded, last_hidden)

        s_t = hidden[0][-1].unsqueeze(0)

        alpha, context = self.attention(encoder_outputs, s_t, inp_mask)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden


class Seq2SeqmitAttn(nn.Module):
    """
    Sequence to sequence model with Attention
    """

    def __init__(self, hidden_size, max_r, n_words, b_size, emb_dim, sos_tok, eos_tok, itos, gpu=False, lr=0.01,
                 train_emb=True,
                 n_layers=1, clip=2.0, pretrained_emb=None, dropout=0.0, emb_drop=0.0, teacher_forcing_ratio=0.0,
                 use_entity_loss = False, entities_property=None):
        super(Seq2SeqmitAttn, self).__init__()
        self.name = "VanillaSeq2Seq"
        self.input_size = n_words
        self.output_size = n_words
        self.hidden_size = hidden_size
        self.max_r = max_r  ## max response len
        self.lr = lr
        self.emb_dim = emb_dim
        self.decoder_learning_ratio = 5.0
        self.n_layers = n_layers
        self.dropout = dropout
        self.emb_drop = emb_drop
        self.b_size = b_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.sos_tok = sos_tok
        self.eos_tok = eos_tok
        self.itos = itos
        self.clip = clip
        self.use_cuda = gpu
        self.use_entity_loss=use_entity_loss
        self.entities_p=entities_property
        # Common embedding for both encoder and decoder
        self.embedding = nn.Embedding(self.output_size, self.emb_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)
            # if train_emb == False:
            #     self.embedding.weight.requires_grad = False

        # Use single RNN for both encoder and decoder
        # self.rnn = nn.LSTM(emb_dim, hidden_size, n_layers, dropout=dropout)
        # initializing the model
        self.encoder = EncoderRNN(self.n_layers, self.emb_dim, self.hidden_size, self.b_size, self.output_size,
                                  gpu=self.use_cuda)
        self.decoder = Decoder(self.hidden_size, self.emb_dim, self.output_size)

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.embedding = self.embedding.cuda()
            # self.rnn = self.rnn.cuda()

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr * self.decoder_learning_ratio)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = 0
        self.print_every = 1

    def train_batch(self, input_batch, out_batch, input_mask, target_mask,
                    input_length=None, output_length=None, target_kb_mask=None,kb=None):

        self.encoder.train(True)
        self.decoder.train(True)
        self.embedding.train(True)

        if self.use_cuda:
            input_batch=input_batch.cuda()
            out_batch=out_batch.cuda()
            input_mask=input_mask.cuda()
            target_mask=target_mask.cuda()

        inp_emb = self.embedding(input_batch)
        # print (len(out_batch))
        b_size = input_batch.size(1)
        # print (b_size)
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.optimizer.zero_grad()

        # Run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(inp_emb,input_length)

        # target_len = torch.sum(target_mask, dim=0)
        target_len = out_batch.size(0)
        # print (min(max(target_len), self.max_r))
        max_target_length = min(target_len, self.max_r)
        # print (max_target_length)
        if not isinstance(max_target_length, int):
            max_target_length = int(max_target_length.cpu().numpy()) if self.use_cuda else int(
                max_target_length.numpy())

        # Prepare input and output variables
        if self.use_cuda:
            decoder_input = Variable(torch.Tensor([self.sos_tok] * b_size)).cuda().long()
            all_decoder_outputs_vocab = Variable(torch.zeros(int(max_target_length), b_size, self.output_size)).cuda()
        else:
            decoder_input = Variable(torch.Tensor([self.sos_tok] * b_size)).long()
            all_decoder_outputs_vocab = Variable(torch.zeros(int(max_target_length), b_size, self.output_size))

        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers], encoder_hidden[1][:self.decoder.n_layers])
        # print (decoder_input.type())
        # print (decoder_input.size(), out_batch.size())
        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < self.teacher_forcing_ratio
        decoded_words = Variable(torch.zeros(int(max_target_length), b_size)).cuda() if self.use_cuda else \
            Variable(torch.zeros(int(max_target_length), b_size))
        # provide data to decoder
        # if use_teacher_forcing:
        if 1:
            for t in range(max_target_length):
                inp_emb_d = self.embedding(decoder_input)
                decoder_vocab, decoder_hidden = self.decoder(inp_emb_d, decoder_hidden, encoder_outputs, input_mask)
                all_decoder_outputs_vocab[t] = decoder_vocab
                decoder_input = out_batch[t].long()  # Next input is current target



        # print (all_decoder_outputs_vocab.size(), out_batch.size())
        # out_batch = out_batch.transpose(0, 1).contiguous
        target_mask = target_mask.transpose(0, 1).contiguous()
        # print (all_decoder_outputs_vocab.size(), out_batch.size(), target_mask.size())
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),  # -> B x S X VOCAB
            out_batch.transpose(0, 1).contiguous(),  # -> B x S
            target_mask
        )
        loss = loss_Vocab

        if self.use_entity_loss and target_kb_mask is not None:
            entity_loss=compute_ent_loss(self.embedding,
                                         all_decoder_outputs_vocab.contiguous(),
                             out_batch.transpose(0,1).contiguous(),
                             target_kb_mask.transpose(0,1).contiguous(),
                             target_mask)

            loss = loss + entity_loss
        loss.backward()

        # clip gradient
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.optimizer.step()

        self.loss += loss.item()

    def evaluate_batch(self, input_batch, out_batch, input_mask, target_mask, target_kb_mask=None, kb=None):
        """
        evaluating batch
        :param input_batch:
        :param out_batch:
        :param input_mask:
        :param target_mask:
        :return:
        """
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        self.embedding.train(False)

        if self.use_cuda:
            input_batch = input_batch.cuda()
            out_batch = out_batch.cuda()
            input_mask = input_mask.cuda()
            target_mask = target_mask.cuda()

        inp_emb = self.embedding(input_batch)
        # output decoder words

        encoder_outputs, encoder_hidden = self.encoder(inp_emb)
        b_size = inp_emb.size(1)
        # target_len = torch.sum(target_mask, dim=0)
        target_len = out_batch.size(0)
        # print (min(max(target_len), self.max_r))
        max_target_length = (min(target_len, self.max_r))
        # print (max_target_length)
        if not isinstance(max_target_length, int):
            max_target_length = int(max_target_length.cpu().numpy()) if self.use_cuda else int(
                max_target_length.numpy())

        # Prepare input and output variables
        if self.use_cuda:
            decoder_input = Variable(torch.Tensor([self.sos_tok] * b_size)).long().cuda()
            all_decoder_outputs_vocab = Variable(torch.zeros(int(max_target_length), b_size, self.output_size)).cuda()
        else:
            decoder_input = Variable(torch.Tensor([self.sos_tok] * b_size)).long()
            all_decoder_outputs_vocab = Variable(torch.zeros(int(max_target_length), b_size, self.output_size))

        decoded_words = Variable(torch.zeros(int(max_target_length), b_size)).cuda() if self.use_cuda else \
            Variable(torch.zeros(int(max_target_length), b_size))
        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers], encoder_hidden[1][:self.decoder.n_layers])

        # provide data to decoder
        for t in range(max_target_length):
            # print (decoder_input)
            inp_emb_d = self.embedding(decoder_input)
            # print (inp_emb_d.size())
            # print (decoder_input.size())
            decoder_vocab, decoder_hidden = self.decoder(inp_emb_d, decoder_hidden, encoder_outputs, input_mask)

            all_decoder_outputs_vocab[t] = decoder_vocab
            topv, topi = decoder_vocab.data.topk(1)  # get prediction from decoder
            masked_topi= topi * target_kb_mask[t]
            for i in range(self.b_size):
                topi[i] = self.check_entity(topi[i].item(), kb[i])
            decoder_input = Variable(topi.view(-1))  # use this in the next time-steps
            decoded_words[t] = (topi.view(-1))


        target_mask = target_mask.transpose(0, 1).contiguous()

        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),  # -> B x S X VOCAB
            out_batch.transpose(0, 1).contiguous(),  # -> B x S
            target_mask
        )

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        self.embedding.train(True)

        return decoded_words, loss_Vocab.item()

    def evaluate_model(self, data, valid=False, test=False):

        if test:
            batches = data.getTestingBatch(self.b_size)
        elif valid:
            batches = data.getBatches(self.b_size, valid=True, transpose=False)
        else:
            batches = data.getBatches(self.b_size, test=True, transpose=False)

        all_predicted = []
        target_batches = []
        individual_metric = []

        for batch in batches:
            input_batch = Variable(torch.LongTensor(batch.encoderSeqs)).transpose(0, 1)
            target_batch = Variable(torch.LongTensor(batch.targetSeqs)).transpose(0, 1)
            input_batch_mask = Variable(torch.FloatTensor(batch.encoderMaskSeqs)).transpose(0, 1)
            target_batch_mask = Variable(torch.FloatTensor(batch.decoderMaskSeqs)).transpose(0, 1)
            target_kb_mask = Variable(torch.LongTensor(batch.targetKbMask)).transpose(0, 1)
            kb = batch.kb_inputs
            decoded_words, loss_Vocab = self.evaluate_batch(input_batch, target_batch, input_batch_mask,
                                                            target_batch_mask,
                                                            target_kb_mask=target_kb_mask, kb=kb)

            batch_predictions = decoded_words.transpose(0, 1)

            batch_metric_score = 0
            for i, sen in enumerate(batch_predictions):
                predicted = data.sequence2str(sen.cpu().numpy(), clean=True)
                reference = data.sequence2str(batch.targetSeqs[i], clean=True)
                batch_metric_score += nltk.translate.bleu_score.sentence_bleu([reference], predicted)

            batch_metric_score = batch_metric_score / self.b_size

            all_predicted.append(batch_predictions)
            target_batches.append(batch.targetSeqs)
            individual_metric.append(batch_metric_score)

        candidates, references = data.get_candidates(target_batches, all_predicted)

        global_metric_score = nltk.translate.bleu_score.corpus_bleu(references, candidates)

        candidates2, references2 = data.get_candidates(target_batches, all_predicted, True)

        moses_multi_bleu_score = moses_multi_bleu(candidates2, references2, True,
                                                  os.path.join("trained_model", self.__class__.__name__))

        return global_metric_score, individual_metric, moses_multi_bleu_score, loss_Vocab

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        self.print_every += 1
        return 'L:{:.2f}'.format(print_loss_avg)

    def check_entity(self, word, kb):

        if word in self.entities_p.keys() and len(kb)>1:
            for triple in kb:
                if word == triple[0]:
                    return word
                if word == triple[2]:
                    return word
            for triple in kb:
                if self.entities_p[word] == triple[1]:
                    return triple[2]
        else:
            return word

class Seq2SeqAttnmitIntent(nn.Module):
    """
        Sequence to sequence model with Luong Attention
        """
    def __init__(self,  attn_model, hidden_size, input_size, output_size, batch_size, sos_tok, eso_tok, n_layers=1,
                 dropout=0.1, intent_size=3, intent_loss_coff=0.9, lr=0.001, decoder_learning_ratio=5.0,
                 pretrained_emb=None, train_emb=False, clip=50.0, teacher_forcing_ratio=1, gpu=False,
                 use_entity_loss=False):
        super(Seq2SeqAttnmitIntent, self).__init__()

        self.name = "LuongSeq2Seq"

        self.intent_size = intent_size
        self.intent_coff = intent_loss_coff

        self.input_size = input_size
        self.output_size = output_size
        self.use_entity_loss=use_entity_loss
        self.emb_dim = hidden_size
        self.hidden_size = hidden_size

        self.lr = lr
        self.clip = clip
        self.decoder_learning_ratio = 5.0
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.sos_tok = sos_tok
        # self.eos_tok = eos_tok
        # self.itos = itos
        # self.clip = clip
        self.use_cuda = gpu


        # Common embedding for both encoder and decoder
        self.embedding = nn.Embedding(self.output_size, self.emb_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)
            # if train_emb == False:
            #     self.embedding.weight.requires_grad = False

        self.encoder = LuongEncoderRNN(self.input_size, hidden_size,self.emb_dim,self.batch_size,
                                       self.n_layers, dropout=dropout, gpu=self.use_cuda)
        self.decoder = LuongAttnDecoderRNN(attn_model, hidden_size,self.emb_dim, self.output_size,
                                           self.batch_size, self.n_layers,
                                           intent_size=self.intent_size, dropout=dropout,
                                           use_cuda=self.use_cuda)

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.embedding=self.embedding.cuda()

        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr * decoder_learning_ratio)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.plot_every = 20
        self.evaluate_every = 20
        self.loss = 0

    def train_batch(self, input_batch, out_batch, input_mask, target_mask, input_length=None,
                    output_length=None, intent_batch=None,target_kb_mask=None,kb=None):

        self.encoder.train(True)
        self.decoder.train(True)
        self.embedding.train(True)

        intent_output = torch.LongTensor(intent_batch)

        if self.use_cuda:
            input_batch=input_batch.cuda()
            out_batch=out_batch.cuda()
            input_mask=input_mask.cuda()
            target_mask=target_mask.cuda()
            intent_output=intent_output.cuda()

        inp_emb = self.embedding(input_batch)

        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.optimizer.zero_grad()

        # Run words through encoder
        #input_len = torch.sum(input_mask, dim=0)
        encoder_outputs, encoder_hidden = self.encoder(inp_emb, input_length)
        # Prepare input and output variables

        max_target_length = out_batch.shape[0]

        decoder_input = Variable(torch.LongTensor([[self.sos_tok] * self.batch_size])).transpose(0, 1)
        #     print('decoder_input', decoder_input.size())
        decoder_context = encoder_outputs[-1]
        decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

        all_decoder_outputs = Variable(torch.zeros(max_target_length, self.batch_size, self.output_size))

        if not isinstance(max_target_length, int):
            max_target_length = int(max_target_length.cpu().numpy()) if self.use_cuda else int(
                max_target_length.numpy())

        # Prepare input and output variables
        if self.use_cuda:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()
            # decoder_context = decoder_context.cuda()

        # provide data to decoder
        # if use_teacher_forcing:
        if 1:
            for t in range(max_target_length):
                inp_emb_d = self.embedding(decoder_input)
                if t == 0:
                    decoder_output, decoder_context, decoder_hidden, decoder_attn, intent_score = self.decoder(
                        inp_emb_d, decoder_context, decoder_hidden, encoder_outputs, input_mask, intent_batch=True
                    )
                else:
                    decoder_output, decoder_context, decoder_hidden, decoder_attn, _ = self.decoder(
                        inp_emb_d, decoder_context, decoder_hidden, encoder_outputs, input_mask)

                all_decoder_outputs[t] = decoder_output
                decoder_input = out_batch[t]

        # print (all_decoder_outputs_vocab.size(), out_batch.size())
        # out_batch = out_batch.transpose(0, 1).contiguous
        target_mask = target_mask.transpose(0, 1).contiguous()
        # print (all_decoder_outputs_vocab.size(), out_batch.size(), target_mask.size())

        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),  # -> B x S X VOCAB
            out_batch.transpose(0, 1).contiguous(),  # -> B x S
            target_mask
        )

        loss_function_2 = nn.CrossEntropyLoss()

        intent_loss = loss_function_2(intent_score, intent_output)
        loss = loss + intent_loss

        if self.use_entity_loss and target_kb_mask is not None:
            entity_loss = compute_ent_loss(self.embedding,
                                           all_decoder_outputs.contiguous(),
                                           out_batch.transpose(0, 1).contiguous(),
                                           target_kb_mask.transpose(0, 1).contiguous(),
                                           target_mask)

            loss = loss + entity_loss

        #intent_loss.backward(retain_graph=True)
        loss.backward()

        # clip gradient
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.optimizer.step()

        self.loss += loss.item()

    def evaluate_batch(self, input_batch, out_batch, input_mask, target_mask, input_length=None,
                       output_length=None, intent_batch=None):

        if self.use_cuda:
            input_batch = input_batch.cuda()
            input_mask= input_mask.cuda()
            target_mask=target_mask.cuda()
            out_batch=out_batch.cuda()

        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        self.embedding.train(False)

        inp_emb = self.embedding(input_batch)

        # Run through encoder
        encoder_outputs, encoder_hidden = self.encoder(inp_emb, input_length)

        # Create starting vectors for decoder
        decoder_input = Variable(torch.LongTensor([[self.sos_tok] * self.batch_size])).transpose(0, 1)  # SOS

        decoder_context = encoder_outputs[-1]  # Variable(torch.zeros(batch_size, decoder.hidden_size))
        decoder_hidden = encoder_hidden  # Use last (forward) hidden state from encoder

        decoder_maxlength =out_batch.size(0)# max(max(output_length), input_batch.size(0))

        all_decoder_predictions = Variable(torch.zeros(decoder_maxlength, self.batch_size))
        all_decoder_outputs_vocab = Variable(torch.zeros(int(decoder_maxlength), self.batch_size, self.output_size))
        if self.use_cuda:
            decoder_input = decoder_input.cuda()
            # decoder_context = decoder_context.cuda()
            # decoder_hidden = decoder_hidden.cuda()
            all_decoder_outputs_vocab=all_decoder_outputs_vocab.cuda()
            all_decoder_predictions = all_decoder_predictions.cuda()

        # Store output words and attention states
        decoded_words = []

        # Run through decoder
        for di in range(decoder_maxlength):
            inp_emb_d = self.embedding(decoder_input)
            if di == 0:
                decoder_output, decoder_context, decoder_hidden, decoder_attention, intent_scores = self.decoder(
                    inp_emb_d, decoder_context, decoder_hidden, encoder_outputs, input_mask, intent_batch=True
                )
                v, i = intent_scores.data.topk(1)
                intent_pred = i
            else:
                decoder_output, decoder_context, decoder_hidden, decoder_attention, _ = self.decoder(
                    inp_emb_d, decoder_context, decoder_hidden, encoder_outputs, input_mask
                )

            all_decoder_outputs_vocab[di] = decoder_output
            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)

            all_decoder_predictions[di] = topi.squeeze(1)

            # Next input is chosen word
            decoder_input = topi
            if self.use_cuda: decoder_input = decoder_input.cuda()


       # return all_decoder_predictions, intent_pred

        target_mask = target_mask.transpose(0, 1).contiguous()

        loss_Vocab = masked_cross_entropy(
        all_decoder_outputs_vocab.transpose(0, 1).contiguous(),  # -> B x S X VOCAB
        out_batch.transpose(0, 1).contiguous(),  # -> B x S
        target_mask
        )

      # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        self.embedding.train(True)

        return all_decoder_predictions, intent_pred, loss_Vocab.item()

    def evaluate_model(self, data, valid=False, test=False):

        if test:
            batches = data.getTestingBatch(self.batch_size)
            output_file = open(os.path.join(os.path.join("trained_model", self.__class__.__name__), "output_file.txt"),
                               "w")
        elif valid:
            batches = data.getBatches(self.batch_size, valid=True, transpose=False)
        else:
            batches = data.getBatches(self.batch_size, test=True, transpose=False)
            output_file = open(os.path.join(os.path.join("trained_model", self.__class__.__name__), "output_file.txt"),
                               "w")

        all_predicted = []
        target_batches = []
        individual_metric = []
        eval_loss=0

        for batch in batches:
            input_batch = Variable(torch.LongTensor(batch.encoderSeqs)).transpose(0, 1)
            target_batch = Variable(torch.LongTensor(batch.targetSeqs)).transpose(0, 1)
            input_batch_mask = Variable(torch.FloatTensor(batch.encoderMaskSeqs)).transpose(0, 1)
            target_batch_mask = Variable(torch.FloatTensor(batch.decoderMaskSeqs)).transpose(0, 1)

            decoded_words, intent, loss = self.evaluate_batch(input_batch, target_batch, input_batch_mask, target_batch_mask,
                                                batch.encoderSeqsLen, batch.decoderSeqsLen)

            eval_loss += loss
            batch_predictions = decoded_words.transpose(0, 1)

            batch_metric_score = 0
            for i, sen in enumerate(batch_predictions):
                predicted = data.sequence2str(sen.cpu().numpy(), clean=True)
                reference = data.sequence2str(batch.targetSeqs[i], clean=True)
                batch_metric_score += nltk.translate.bleu_score.sentence_bleu([reference], predicted)

                if not valid:
                    output_file.write("\n"+"Input : " +
                                                data.sequence2str(batch.encoderSeqs[i], clean=True))

                    output_file.write("\n"+"Predicted : " +
                                      data.sequence2str(batch_predictions[i].cpu().numpy(), clean=True) +
                                      ", intent:" + data.id2intent[intent[i][0].item()]
                                      )

                    output_file.write("\n"+"Target : " +
                                      data.sequence2str(batch.targetSeqs[i], clean=True) +
                                      ", intent:" + data.id2intent[batch.seqIntent[i]])
                    output_file.write("\n")
                    output_file.flush()

            batch_metric_score = batch_metric_score / self.batch_size

            all_predicted.append(batch_predictions)
            target_batches.append(batch.targetSeqs)
            individual_metric.append(batch_metric_score)

        candidates, references = data.get_candidates(target_batches, all_predicted)

        global_metric_score = nltk.translate.bleu_score.corpus_bleu(references, candidates)

        candidates2, references2 = data.get_candidates(target_batches, all_predicted, True)
        if not valid and not test:
            moses_multi_bleu_score = moses_multi_bleu(candidates2,references2, True, os.path.join("trained_model", self.__class__.__name__))
        else:
            moses_multi_bleu_score = moses_multi_bleu(candidates2, references2, True)

        return global_metric_score, individual_metric, moses_multi_bleu_score, eval_loss



