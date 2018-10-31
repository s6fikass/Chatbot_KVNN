from tqdm import tqdm
import random
import time
import math
import socket
import os
import argparse
import numpy as np

from corpus.textdata import TextData
from model.seq2seq_model import EncoderRNN, LuongAttnDecoderRNN, Seq2SeqmitAttn, Seq2SeqLuongAttn  # KVEncoderRNN, KVAttnDecoderRNN,

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional
from torch.autograd import Variable


from train import train

from evaluate import evaluate_model, evaluate_randomly, evaluate_sample

USE_CUDA = False
hostname = socket.gethostname()


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


def main(args):

    # get data
    train_file = 'data/kvret_train_public.json'
    valid_file = 'data/kvret_dev_public.json'
    test_file = 'data/kvret_test_public.json'
    textdata = TextData(train_file, valid_file, test_file)
    args.data = textdata

    print('Datasets Loaded.')

    print('Compiling Model.')

    # Configure models
    attn_model = 'dot'
    n_layers = 1
    dropout = 0.0
    hidden_size = 200

    # Configure training/optimization
    n_epochs = args.epochs
    epoch = 0
    plot_every = 20
    evaluate_every = 10
    avg_best_metric = 0
    save_every = 5

    # Initialize models
    if args.intent:
        model = Seq2SeqLuongAttn(attn_model, hidden_size,textdata.getVocabularySize(), textdata.getVocabularySize(),
                                 args.batch_size, textdata.word2id['<go>'], textdata.word2id['<eos>'], gpu=args.cuda)
    else:
        model = Seq2SeqmitAttn(hidden_size, textdata.getTargetMaxLength(), textdata.getVocabularySize(),
                               args.batch_size, hidden_size, textdata.word2id['<go>'], textdata.word2id['<eos>'],
                               None, gpu=args.cuda, lr=0.001, train_emb=False,
                               n_layers=1, clip=2.0, pretrained_emb=None, dropout=0.0, emb_drop=0.0,
                               teacher_forcing_ratio=0.0)

    directory = os.path.join("trained_model", model.__class__.__name__)

    if not os.path.exists(directory):
        os.makedirs(directory)

    if args.loadFilename:
        if args.cuda:
            model.load_state_dict(torch.load(os.path.join(directory, '{}.bin'.format(args.loadFilename))))
        else:
            model.load_state_dict(torch.load(os.path.join(directory, '{}.bin'.format(args.loadFilename)),
                                                          map_location=lambda storage, loc: storage))

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

    print('Model Compiled.')
    print('Training. Ctrl+C to end early.')

    if args.val:
        global_metric_score, individual_metric, moses_multi_bleu_score = \
            model.evaluate_model(args, textdata)
        print("Model Bleu using corpus bleu: ", global_metric_score)
        print("Model Bleu using sentence bleu: ", sum(individual_metric)/len(individual_metric))
        print("Model Bleu using moses_multi_bleu_score :", moses_multi_bleu_score)
    else:
        total_loss = 0
        while epoch < n_epochs:
            epoch += 1
            # steps_done = 0

            if args.test:
                batches = textdata.getTestingBatch(args.batch_size)
            else:
                batches = textdata.getBatches(args.batch_size, transpose=False)

            # steps_per_epoch = len(batches)
            try:

                epoch_ec = 0
                epoch_dc = 0

                for current_batch in tqdm(batches, desc='Processing batches'):

                    kb_batch=current_batch.kb_inputs

                    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
                    target_lengths = current_batch.decoderSeqsLen
                    input_lengths = current_batch.encoderSeqsLen

                    input_batch = Variable(torch.LongTensor(current_batch.encoderSeqs)).transpose(0, 1)
                    target_batch = Variable(torch.LongTensor(current_batch.targetSeqs)).transpose(0, 1)
                    input_batch_mask = Variable(torch.FloatTensor(current_batch.encoderMaskSeqs)).transpose(0, 1)
                    target_batch_mask = Variable(torch.FloatTensor(current_batch.decoderMaskSeqs)).transpose(0, 1)

                    # Train Model
                    if args.intent:
                        model.train_batch(input_batch, target_batch, input_batch_mask, target_batch_mask,
                                          input_lengths,target_lengths)
                    else:
                        model.train_batch(input_batch, target_batch, input_batch_mask, target_batch_mask)

                # Keep track of loss
                print_loss_total += model.loss
                plot_loss_total += model.loss
                # eca += epoch_ec
                # dca += epoch_dc

                epoch_loss = model.loss-total_loss
                total_loss = model.loss
                print(epoch, epoch_loss, "epoch-loss")

                # if epoch == 1:
                #     evaluate_randomly(args, textdata, encoder, decoder)
                #     continue
                #
                if epoch % evaluate_every == 0:
                    print_loss_avg = print_loss_total / evaluate_every
                #     print_loss_total = 0
                    print_summary = '%s (%d %d%%) %.4f' % (
                            time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
                    print(print_summary)

                    global_metric_score, individual_metric, moses_multi_bleu_score = \
                        model.evaluate_model(textdata, test=True)

                    print("Model Bleu using corpus bleu: ", global_metric_score)
                    print("Model Bleu using sentence bleu: ", sum(individual_metric) / len(individual_metric))
                    print("Model Bleu using moses_multi_bleu_score :", moses_multi_bleu_score)
                    bleu =moses_multi_bleu_score
                    max(global_metric_score, sum(individual_metric) / len(individual_metric),
                               moses_multi_bleu_score/100)

                    if bleu > avg_best_metric:
                        avg_best_metric = bleu
                        cnt = 0
                    else:
                        cnt += 1

                    if cnt == 10: break

                # if epoch % plot_every == 0:
                #
                #     plot_loss_avg = plot_loss_total / plot_every
                #     plot_losses.append(plot_loss_avg)
                #     plot_loss_total = 0
                #     #
                #     # # TODO: Running average helper
                #     ecs.append(eca / plot_every)
                #     dcs.append(dca / plot_every)
                #     # ecs_win = 'encoder grad (%s)' % hostname
                #     # dcs_win = 'decoder grad (%s)' % hostname
                #     # print(ecs)
                #     # print(dcs)
                #     # # vis.line(np.array(ecs), win=ecs_win, opts={'title': ecs_win})
                #     # # vis.line(np.array(dcs), win=dcs_win, opts={'title': dcs_win})
                #     eca = 0
                #     dca = 0
                #
                # if epoch % save_every == 0:
                #     directory = os.path.join("trained_model", decoder.__class__.__name__,
                #                                  '{}-{}_{}'.format(n_layers, epoch, hidden_size))
                #
                #     if not os.path.exists(directory):
                #         os.makedirs(directory)
                #     torch.save({
                #             'epoch': epoch,
                #             'en': encoder.state_dict(),
                #             'de': decoder.state_dict(),
                #             'en_opt': encoder_optimizer.state_dict(),
                #             'de_opt': decoder_optimizer.state_dict(),
                #             'loss': print_loss_total,
                #             'plot_losses':plot_losses,
                #             'ecs':ecs,
                #             'dcs':dca,
                #         }, os.path.join(directory, '{}_{}.tar'.format(epoch, "model_glove_{}".format(args.glove))))
                break
            except KeyboardInterrupt as e:
                print('Model training stopped early.')
                break

        # model.save_weights("model_weights_nkbb.hdf5")
        print('Model training complete.')

        global_metric_score, individual_metric, moses_multi_bleu_score = \
            model.evaluate_model(textdata)
        print("Test Model Bleu using corpus bleu: ", global_metric_score)
        print("Test Model Bleu using sentence bleu: ", sum(individual_metric) / len(individual_metric))
        print("Test Model Bleu using moses_multi_bleu_score :", moses_multi_bleu_score)

        print('Saving Model.')
        torch.save(model.state_dict(), os.path.join(directory,'{}.bin'.format(epoch)))


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
                            required=False, default=10, type=int)

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

    named_args.add_argument('-glove', '--glove', metavar='|',
                            help="""to use glove embeddings """,
                            required=False, default=False, type=bool)

    named_args.add_argument('-intent', '--intent', metavar='|',
                            help="""Joint learning based on intent """,
                            required=False, default=False, type=bool)

    named_args.add_argument('-test', '--test', metavar='|',
                            help="""test the model on one batch and evaluate on the same """,
                            required=False, default=False, type=bool)

    args = parser.parse_args()
    #args.model="KVSeq2Seq"
    #args.val=True
    if args.cuda:
        USE_CUDA = True

    # try:
    #     torch.LongTensor([1,2]).cuda()
    #     USE_CUDA = True
    # except:
    #     print("no cuda")
    #     USE_CUDA = False
    main(args)






