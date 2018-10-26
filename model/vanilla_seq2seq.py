#imports
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import optim
from torch.autograd import Variable
from util.utils import masked_cross_entropy

class Seq2SeqmitAttn(nn.Module):
    """
    Sequence to sequence model with Attention
    """
    def __init__(self, hidden_size, max_r, n_words, b_size, emb_dim, sos_tok, eos_tok, itos, gpu=False, lr=0.01, train_emb=False,
                 n_layers=1, clip=2.0, pretrained_emb=None, dropout=0.1, emb_drop=0.2, teacher_forcing_ratio=0.0):
        super(Seq2SeqmitAttn, self).__init__()
        self.name = "VanillaSeq2Seq"
        self.input_size = n_words
        self.output_size = n_words
        self.hidden_size = hidden_size
        self.max_r = max_r ## max response len
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
        # Common embedding for both encoder and decoder
        self.embedding = nn.Embedding(self.output_size, self.emb_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)
        if train_emb == False:
            self.embedding.weight.requires_grad = False

        # Use single RNN for both encoder and decoder
        #self.rnn = nn.LSTM(emb_dim, hidden_size, n_layers, dropout=dropout)
        # initializing the model
        self.encoder = EncoderRNN(self.n_layers, self.emb_dim, self.hidden_size, self.b_size, self.output_size,
                                 gpu=self.use_cuda)
        self.decoder = Decoder(self.hidden_size, self.emb_dim, self.output_size)

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.embedding = self.embedding.cuda()
            #self.rnn = self.rnn.cuda()

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr * self.decoder_learning_ratio)

        self.loss = 0
        self.print_every = 1

    def train_batch(self, input_batch, out_batch, input_mask, target_mask):

        self.encoder.train(True)
        self.decoder.train(True)
        self.embedding.train(True)

        inp_emb = self.embedding(input_batch)
        #print (len(out_batch))
        b_size = input_batch.size(1)
        #print (b_size)
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss_Vocab,loss_Ptr,loss_Gate = 0,0,0
        # Run words through encoder
        input_len = torch.sum(input_mask, dim=0)
        encoder_outputs, encoder_hidden = self.encoder(inp_emb)

        #target_len = torch.sum(target_mask, dim=0)
        target_len = out_batch.size(0)
        #print (min(max(target_len), self.max_r))
        max_target_length = min(target_len, self.max_r)
        #print (max_target_length)
        if not isinstance(max_target_length, int):
            max_target_length = int(max_target_length.cpu().numpy()) if self.use_cuda else int(max_target_length.numpy())

        # Prepare input and output variables
        if self.use_cuda:
            decoder_input = Variable(torch.Tensor([self.sos_tok] * b_size)).cuda().long()
            all_decoder_outputs_vocab = Variable(torch.zeros(int(max_target_length), b_size, self.output_size)).cuda()
        else:
            decoder_input = Variable(torch.Tensor([self.sos_tok] * b_size)).long()
            all_decoder_outputs_vocab = Variable(torch.zeros(int(max_target_length), b_size, self.output_size))

        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers], encoder_hidden[1][:self.decoder.n_layers])
        #print (decoder_input.type())
        #print (decoder_input.size(), out_batch.size())
        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < self.teacher_forcing_ratio

        # provide data to decoder
        #if use_teacher_forcing:
        if 1:
            for t in range(max_target_length):
                inp_emb_d = self.embedding(decoder_input)
                decoder_vocab, decoder_hidden = self.decoder(inp_emb_d, decoder_hidden, encoder_outputs, input_mask)
                all_decoder_outputs_vocab[t] = decoder_vocab
                decoder_input = out_batch[t].long() # Next input is current target
        else:
            for t in range(max_target_length):
                inp_emb_d = self.embedding(decoder_input)
                decoder_vocab, decoder_hidden = self.decoder(inp_emb_d, decoder_hidden, encoder_outputs, input_mask)
                all_decoder_outputs_vocab[t] = decoder_vocab
                topv, topi = decoder_vocab.data.topk(1) # get prediction from decoder
                decoder_input = Variable(topi.view(-1)) # use this in the next time-steps

        #print (all_decoder_outputs_vocab.size(), out_batch.size())
        #out_batch = out_batch.transpose(0, 1).contiguous
        target_mask = target_mask.transpose(0,1).contiguous()
        #print (all_decoder_outputs_vocab.size(), out_batch.size(), target_mask.size())
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), # -> B x S X VOCAB
            out_batch.transpose(0, 1).contiguous(), # -> B x S
            target_mask
        )

        loss = loss_Vocab
        loss.backward()

        # clip gradient
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)


        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.data[0]


    def evaluate_batch(self, input_batch, out_batch, input_mask, target_mask):
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

        inp_emb = self.embedding(input_batch)
        # output decoder words



        encoder_outputs, encoder_hidden = self.encoder(inp_emb)
        b_size = inp_emb.size(1)
        #target_len = torch.sum(target_mask, dim=0)
        target_len = out_batch.size(0)
        #print (min(max(target_len), self.max_r))
        max_target_length = (min(target_len, self.max_r))
        #print (max_target_length)
        if not isinstance(max_target_length, int):
            max_target_length = int(max_target_length.cpu().numpy()) if self.use_cuda else int(max_target_length.numpy())


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
            #print (decoder_input)
            inp_emb_d = self.embedding(decoder_input)
            #print (inp_emb_d.size())
            #print (decoder_input.size())
            decoder_vocab, decoder_hidden = self.decoder(inp_emb_d, decoder_hidden, encoder_outputs, input_mask)
            # if decoder_vocab.size(0) < self.b_size:
            #     if self.use_cuda:
            #         decoder_vocab = torch.cat([decoder_vocab, torch.zeros(b_size-decoder_vocab.size(0), self.output_size).cuda()])
            #     else:
            #         decoder_vocab = torch.cat([decoder_vocab, torch.zeros(b_size-decoder_vocab.size(0), self.output_size)])
            all_decoder_outputs_vocab[t] = decoder_vocab
            topv, topi = decoder_vocab.data.topk(1) # get prediction from decoder
            decoder_input = Variable(topi.view(-1)) # use this in the next time-steps
            decoded_words[t] = (topi.view(-1))
            #decoded_words.append(['<EOS>' if ni == self.eos_tok else self.itos(ni) for ni in topi.view(-1)])
        #print (all_decoder_outputs_vocab.size(), out_batch.size())
        #out_batch = out_batch.transpose(0, 1).contiguous
        target_mask = target_mask.transpose(0,1).contiguous()

        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), # -> B x S X VOCAB
            out_batch.transpose(0, 1).contiguous(), # -> B x S
            target_mask
        )

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        self.embedding.train(True)

        return decoded_words, loss_Vocab

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        self.print_every += 1
        return 'L:{:.2f}'.format(print_loss_avg)



class EncoderRNN(nn.Module):
    """
    Encoder RNN module
    """
    def __init__(self, input_size, emb_size, hidden_size, b_size, vocab_size, n_layers=1, dropout=0.1, emb_drop=0.2,
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
        #self.rnn = rnn

    def init_weights(self, b_size):
        #intiialize hidden weights
        c0 = Variable(torch.zeros(self.n_layers, b_size, self.hidden_size))
        h0 = Variable(torch.zeros(self.n_layers, b_size, self.hidden_size))

        if self.gpu:
            c0 = c0.cuda()
            h0 = h0.cuda()

        return h0, c0

    def forward(self, inp_emb, input_lengths=None):
        #input_q = S X B X EMB
        #embedded = self.embedding(input_q)
        embedded = self.embedding_dropout(inp_emb)
        hidden = self.init_weights(inp_emb.size(1))

        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)

        outputs, hidden = self.rnn(embedded, hidden) # outputs = S X B X n_layers*H, hidden = 2 * [1 X B X H]
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        return outputs, hidden

class Attention(nn.Module):
    """
    Attention mechanism (Luong)
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        #weights
        self.W_h = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.epsilon = 1e-10

    def forward(self, encoder_outputs, decoder_hidden, inp_mask):
        seq_len = encoder_outputs.size(1) # get sequence lengths S
        H = decoder_hidden.repeat(seq_len, 1, 1).transpose(0, 1) # B X S X H
        energy = F.tanh(self.W_h(torch.cat([H, encoder_outputs], 2))) # B X S X H
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B X 1 X H]
        energy = torch.bmm(v,energy).view(-1, seq_len) # [B X T]
        a = F.softmax(energy) * inp_mask.transpose(0, 1) # B X T
        normalization_factor = a.sum(1, keepdim=True)
        a = a / (normalization_factor+self.epsilon) # adding a small offset to avoid nan values

        a = a.unsqueeze(1)
        context = a.bmm(encoder_outputs)

        return a, context

class Decoder(nn.Module):
    """
    Decoder RNN
    """
    def __init__(self, hidden_size, emb_dim, vocab_size, n_layers=1, dropout=0.1):
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
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden
