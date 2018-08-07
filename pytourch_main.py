#from tqdm import tqdm
import unicodedata
import string
import re
import random
import time
#import datetime
import math
import socket
import os
#from json import JSONDecoder

#from reader import Data,Vocabulary
import argparse

from corpus.textdata import TextData
from model.seq2seq_model import KVEncoderRNN, KVAttnDecoderRNN, EncoderRNN, LuongAttnDecoderRNN

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


def masked_cross_entropy(logits, target, target_lengths):
    length = Variable(torch.LongTensor(target_lengths))

    if USE_CUDA:
        length.cuda()

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
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
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


# PAD_token = 1520
#EOS_token = 1522
SOS_token = 1
# EOU_token = 3

teacher_forcing_ratio = 0.5

def train_kb(args, input_batches, target_batches, kb_batch, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion,batch_size,target_lengths, clip = 50.0):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, None)

    # Prepare input and output variables

    decoder_input = Variable(torch.LongTensor([[SOS_token] * batch_size])).transpose(0, 1)
    #     print('decoder_input', decoder_input.size())
    decoder_context = encoder_outputs[-1]
    decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

    max_target_length = target_batches.shape[0]
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio

    # TODO: Get targets working
    if True:
        # Run through decoder one time step at a time
        for t in range(max_target_length):

            decoder_output, decoder_context, decoder_hidden, decoder_attn, kb_attn = decoder(
                decoder_input, kb_batch, decoder_context, decoder_hidden, encoder_outputs
            )
            all_decoder_outputs[t] = decoder_output

            decoder_input = target_batches[t]
            # TODO decoder_input = target_variable[di] # Next target is next input

    loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),  # seq x batch -> batch x seq
            target_batches.transpose(0, 1).contiguous(),  # seq x batch -> batch x seq
        target_lengths
        )

    loss.backward()

    # Clip gradient norm
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item(), ec, dc

def train(args, input_batches, target_batches, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion,batch_size, input_lengths, target_lengths, clip = 50.0):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths)

    # Prepare input and output variables

    decoder_input = Variable(torch.LongTensor([[SOS_token] * batch_size])).transpose(0, 1)
    #     print('decoder_input', decoder_input.size())
    decoder_context = encoder_outputs[-1]
    decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

    max_target_length = target_batches.shape[0]
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio

    # TODO: Get targets working
    if True:
        # Run through decoder one time step at a time
        for t in range(max_target_length):

            decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_context, decoder_hidden, encoder_outputs
            )
            all_decoder_outputs[t] = decoder_output

            decoder_input = target_batches[t]
            # TODO decoder_input = target_variable[di] # Next target is next input

    loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),  # seq x batch -> batch x seq
            target_batches.transpose(0, 1).contiguous(),  # seq x batch -> batch x seq
        target_lengths
        )

    loss.backward()

    # Clip gradient norm
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item(), ec, dc


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def evaluate_sample(vocab,encoder, decoder, input_seqs, input_length=None):

    input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1)
    if input_length is None:
        max_length = input_batches.shape[0]
    else:
        max_length = max(input_length)

    if USE_CUDA:
        input_batches = input_batches.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_length)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_token]))  # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden  # Use last (forward) hidden state from encoder

    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
            decoder_input,decoder_context, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data



        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0].item()

        if ni == vocab.word2id['<eos>']:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(vocab.id2word[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))
        if USE_CUDA: decoder_input = decoder_input.cuda()

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]


def evaluate(vocab,encoder, decoder, input_seqs, input_length=None):

    input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1)
    batch_size = input_batches.shape[1]

    if input_length is None:
        max_length = input_batches.shape[0]
    else:
        max_length = max(input_length)

    if USE_CUDA:
        input_batches = input_batches.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_length)


    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token] * batch_size])).transpose(0, 1)  # SOS

    decoder_context = encoder_outputs[-1]#Variable(torch.zeros(batch_size, decoder.hidden_size))
    decoder_hidden = encoder_hidden  # Use last (forward) hidden state from encoder
    all_decoder_outputs = Variable(torch.zeros(max_length, batch_size))
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []

    # Run through decoder
    for di in range(max_length):

        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
            decoder_input,decoder_context, decoder_hidden, encoder_outputs
        )

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        # for i,k in enumerate(topi):
        #     print(i)
        #     print(k.item())
        #     all_decoder_outputs[i].append
        # ni = topi[0][0].item()
        #
        # if ni == vocab.word2id['<eos>']:
        #     decoded_words.append('<EOS>')
        #     break
        # else:
        #     decoded_words.append(vocab.id2word[ni])

        all_decoder_outputs[di]= topi.squeeze(1)


        # Next input is chosen word
        decoder_input = topi
        if USE_CUDA: decoder_input = decoder_input.cuda()

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return all_decoder_outputs


def evaluate_randomly(data, encoder, decoder):
    #input_sentence, kb, target_sentence = next(data.random_batch(1))

    batch = data.getBatches(1, valid=True, transpose=False)[0]
    input_sentence = batch.encoderSeqs
    target_sentence = batch.decoderSeqs
    evaluate_and_show_attention(data,encoder, decoder, input_sentence, target_sentence, batch.encoderSeqsLen)


def evaluate_model(args, data, encoder, decoder):

    batches = data.getBatches(args.batch_size, transpose=False)
    print(len(batches))

    for batch in batches:
        input_seq = batch.encoderSeqs
        target_seq = batch.decoderSeqs

        evaluate_and_calculate_blue(data, encoder, decoder, input_seq, target_seq, batch.encoderSeqsLen)
        break

def evaluate_and_calculate_blue(data, encoder, decoder, input_sentence, target_sentence, input_length):
    output_words = evaluate(data, encoder, decoder, input_sentence, input_length)

    for sen in output_words.transpose(0,1):
        print(data.sequence2str(sen.numpy(),clean=True))

import io
import torchvision
from PIL import Image
# import visdom
# vis = visdom.Visdom()

def show_plot_visdom():
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    attn_win = 'attention (%s)' % hostname
    # vis.image(torchvision.transforms.ToTensor()(Image.open(buf)), win=attn_win, opts={'title': attn_win})


def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<eos>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #show_plot_visdom()
    plt.draw()
    plt.close()


def evaluate_and_show_attention(vocab, encoder, decoder, input_sentence, target_sentence, input_length):
    output_words, attentions = evaluate_sample(vocab, encoder, decoder, input_sentence, input_length)
    print(output_words)
    output_sentence = ' '.join(output_words)
    input_sentence= vocab.sequence2str(input_sentence[0],clean=True)
    print('>', input_sentence)
    if target_sentence is not None:
        target_sentence= vocab.sequence2str(target_sentence[0])
        print('=', target_sentence)
    print('<', output_sentence)

    show_attention(input_sentence, output_words, attentions)

    # Show input, target, output text in visdom
    win = 'evaluted (%s)' % hostname
    text = '<p>&gt; %s</p><p>= %s</p><p>&lt; %s</p>' % (input_sentence, target_sentence, output_sentence)
    # vis.text(text, win=win, opts={'title': win})

# create a directory if it doesn't already exist
if not os.path.exists('./weights'):
    os.makedirs('./weights/')

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    # # Dataset functions
    # vocab = Vocabulary('data/vocabulary.json',
    #                           padding=args.padding)
    # kb_vocab = Vocabulary('data/vocabulary.json',
    #                           padding=4)
    #
    # print('Loading datasets.')
    #
    #
    # training = Data(args.training_data, vocab,kb_vocab)
    # #validation = Data(args.validation_data, vocab, kb_vocab)
    # training.load()
    # #validation.load()
    # training.transform()
    # training.kb_out()
    # #validation.transform()
    # #validation.kb_out()
    # training_generator = training.random_batch(args.batch_size)

    # get data

    train_file = 'data/kvret_train_public.json'
    valid_file = 'data/kvret_dev_public.json'
    test_file = 'data/kvret_test_public.json'
    model_dir = "pytourch_trained_model"

    textdata= TextData("data/kvret_train_public.json",'data/kvret_dev_public.json','data/kvret_test_public.json')

    print('Datasets Loaded.')
    print('Compiling Model.')

    # Configure models
    attn_model = 'dot'
    n_layers = 2
    dropout = 0.1
    hidden_size = 200

    # Configure training/optimization
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_epochs = args.epochs
    epoch = 0
    plot_every = 20
    print_every = 20
    evaluate_every = 1000


    # Initialize models
    if args.model=="KVSeq2Seq":
        encoder = EncoderRNN(textdata.getVocabularySize(), hidden_size, n_layers, dropout=dropout)
        decoder = KVAttnDecoderRNN(attn_model, hidden_size, textdata.getVocabularySize(), n_layers, dropout=dropout)
    else:
        encoder = EncoderRNN(textdata.getVocabularySize(), hidden_size, n_layers, dropout=dropout)
        decoder = LuongAttnDecoderRNN(attn_model, hidden_size, textdata.getVocabularySize(), n_layers, dropout=dropout)

    if args.loadFilename:
        checkpoint = torch.load(args.loadFilename)
        encoder.load_state_dict(checkpoint['enc'])
        decoder.load_state_dict(checkpoint['dec'])

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    criterion = nn.CrossEntropyLoss()

    if args.loadFilename:
        encoder_optimizer.load_state_dict(checkpoint['enc_opt'])
        decoder_optimizer.load_state_dict(checkpoint['dec_opt'])

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Begin!
    ecs = []
    dcs = []
    eca = 0
    dca = 0

    save_every =500
    print('Model Compiled.')
    print('Training. Ctrl+C to end early.')
    if args.val:
        evaluate_model(args, textdata, encoder, decoder)
    else:

        while epoch < n_epochs:
            epoch += 1
            steps_done = 0
            batches = textdata.getBatches(args.batch_size, transpose=False)
            steps_per_epoch = len(batches)
            try:
                epoch_loss = 0
                epoch_ec = 0
                epoch_dc = 0

                while steps_done < steps_per_epoch:
                #for current_batch in tqdm(batches, desc='Processing batches'):
                    current_batch=batches[steps_done]
                    x = current_batch.encoderSeqs
                    y = current_batch.decoderSeqs

                    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
                    target_lengths = current_batch.decoderSeqsLen
                    input_lengths = current_batch.encoderSeqsLen


                    input_batch = Variable(torch.LongTensor(x)).transpose(0, 1)
                    target_batch = Variable(torch.LongTensor(y)).transpose(0, 1)

                    if args.model == "KVSeq2Seq":
                        kb = current_batch.kb
                        kb_batch = Variable(torch.LongTensor(kb))

                        # Run the train function
                        loss, ec, dc = train_kb(args, input_batch, target_batch, kb_batch,
                                             encoder, decoder,encoder_optimizer, decoder_optimizer,
                                            criterion, args.batch_size, input_lengths, target_lengths
                                                )

                    # Run the train function
                    loss, ec, dc = train(args, input_batch, target_batch, encoder, decoder,
                                         encoder_optimizer, decoder_optimizer, criterion,
                                         args.batch_size, input_lengths, target_lengths
                                         )

                    epoch_loss += loss
                    epoch_ec += ec
                    epoch_dc += dc
                    steps_done += 1

                # Keep track of loss
                print_loss_total += epoch_loss
                plot_loss_total += epoch_loss
                eca += epoch_ec
                dca += epoch_dc

                print( epoch, epoch_loss, "step-epoch-loss")

                if epoch == 1:
                    evaluate_randomly(textdata, encoder, decoder)
                    continue

                if epoch % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print_summary = '%s (%d %d%%) %.4f' % (
                            time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
                    print(print_summary)
                    evaluate_randomly(textdata, encoder, decoder)

                if epoch % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

                    # TODO: Running average helper
                    ecs.append(eca / plot_every)
                    dcs.append(dca / plot_every)
                    ecs_win = 'encoder grad (%s)' % hostname
                    dcs_win = 'decoder grad (%s)' % hostname
                    print(ecs)
                    print(dcs)
                    # vis.line(np.array(ecs), win=ecs_win, opts={'title': ecs_win})
                    # vis.line(np.array(dcs), win=dcs_win, opts={'title': dcs_win})
                    eca = 0
                    dca = 0

                if epoch % save_every == 0:
                        directory = os.path.join("trained_model", decoder.__class__.__name__,
                                                 '{}-{}_{}'.format(n_layers, epoch, hidden_size))
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        torch.save({
                            'epoch': epoch,
                            'en': encoder.state_dict(),
                            'de': decoder.state_dict(),
                            'en_opt': encoder_optimizer.state_dict(),
                            'de_opt': decoder_optimizer.state_dict(),
                            'loss': print_loss_total
                        }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'backup_bidir_model')))

            except KeyboardInterrupt as e:
                print('Model training stopped early.')
                break

        # model.save_weights("model_weights_nkbb.hdf5")
        print('Model training complete.')
        print('Saving Model.')

        directory = os.path.join("trained_model", decoder.__class__.__name__,
                                 '{}'.format(epoch))
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'epoch': epoch,
            'enc': encoder.state_dict(),
            'dec': decoder.state_dict(),
            'enc_opt': encoder_optimizer.state_dict(),
            'dec_opt': decoder_optimizer.state_dict(),
            'loss': print_loss_total
        }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'backup_bidir_model')))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-e', '--epochs', metavar='|',
                            help="""Number of Epochs to Run""",
                            required=False, default=1500, type=int)

    named_args.add_argument('-es', '--embedding', metavar='|',
                            help="""Size of the embedding""",
                            required=False, default=200, type=int)

    named_args.add_argument('-g', '--gpu', metavar='|',
                            help="""GPU to use""",
                            required=False, default='1', type=str)

    named_args.add_argument('-p', '--padding', metavar='|',
                            help="""Amount of padding to use""",
                            required=False, default=20, type=int)

    named_args.add_argument('-t', '--training-data', metavar='|',
                            help="""Location of training data""",
                            required=False, default='./data/train_data.csv')

    named_args.add_argument('-v', '--validation-data', metavar='|',
                            help="""Location of validation data""",
                            required=False, default='./data/val_data.csv')

    named_args.add_argument('-b', '--batch-size', metavar='|',
                            help="""Location of validation data""",
                            required=False, default=100, type=int)

    named_args.add_argument('-tm', '--loadFilename', metavar='|',
                            help="""Location of trained model """,
                            required=False, default=None, type=str)

    named_args.add_argument('-m', '--model', metavar='|',
                            help="""Location of trained model """,
                            required=False, default="Seq2Seq", type=str)

    named_args.add_argument('-vv', '--val', metavar='|',
                            help="""Location of trained model """,
                            required=False, default=False, type=bool)

    named_args.add_argument('-cuda', '--cuda', metavar='|',
                            help="""to use cuda """,
                            required=False, default=False, type=bool)

    args = parser.parse_args()
    #args.model="KVSeq2Seq"
    #args.val=True
    print(args)
    main(args)






