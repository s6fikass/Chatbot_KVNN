


import io

import torch
from torch.autograd import Variable

from util.measures import moses_multi_bleu

import nltk
from sklearn.metrics.pairwise import cosine_distances

#matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import torchvision
# from PIL import Image
# import visdom
# vis = visdom.Visdom()

import socket

SOS_token = 1
teacher_forcing_ratio = 0.5

hostname = socket.gethostname()

def evaluate_sample(args, vocab, encoder, decoder, input_seqs, input_length=None, output_length = None):

    input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1)
    if input_length is None:
        max_length = input_batches.shape[0]
    else:
        max_length = max(input_length)

    if args.cuda:
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

    if args.cuda:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    if output_length is None:
        max_output_length = max_length +2
    else:
        max_output_length = max(output_length)

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_output_length + 1, max_output_length + 1)

    # Run through decoder
    for di in range(max_output_length):
        if args.intent and di == 0:
            decoder_output, decoder_context, decoder_hidden, decoder_attention,intent_scores = decoder(
            decoder_input,decoder_context, decoder_hidden, encoder_outputs, intent_batch=True
            )
        else:
            decoder_output, decoder_context, decoder_hidden, decoder_attention, _ = decoder(
                decoder_input, decoder_context, decoder_hidden, encoder_outputs
            )

        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data



        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0].item()

        if ni == vocab.word2id['<eos>']:
            decoded_words.append('<eos>')
            break
        else:
            decoded_words.append(vocab.id2word[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))
        if args.cuda: decoder_input = decoder_input.cuda()
    if args.intent:
        v,i=intent_scores.data.topk(1)
        decoded_words.append("<"+vocab.id2intent[i[0][0].item()]+">")
    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]


def evaluate(args,encoder, decoder, input_seqs, target_seqs, input_length=None, output_length=None):

    input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1)
    batch_size = input_batches.shape[1]

    if input_length is None:
        max_length = input_batches.shape[0]
    else:
        max_length = max(input_length)

    if args.cuda:
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
    if output_length is None:
        decoder_maxlength=max_length +2
    else:
        decoder_maxlength = max(output_length)

    all_decoder_predictions = Variable(torch.zeros(decoder_maxlength, batch_size))
    #all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))
    if args.cuda:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
        decoder_hidden = decoder_hidden.cuda()
        all_decoder_predictions = all_decoder_predictions.cuda()

    # Store output words and attention states
    decoded_words = []

    # Run through decoder
    for di in range(decoder_maxlength):

        if args.intent and di == 0:
            decoder_output, decoder_context, decoder_hidden, decoder_attention,intent_scores = decoder(
            decoder_input, decoder_context, decoder_hidden, encoder_outputs, intent_batch=True
            )
        else:
            decoder_output, decoder_context, decoder_hidden, decoder_attention, _ = decoder(
                decoder_input, decoder_context, decoder_hidden, encoder_outputs
            )

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)

        all_decoder_predictions[di] = topi.squeeze(1)


        # Next input is chosen word
        decoder_input = topi
        if args.cuda: decoder_input = decoder_input.cuda()

        if args.intent:
            v, i = intent_scores.data.topk(1)
            intent_pred=i


    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return all_decoder_predictions, intent_pred


def evaluate_randomly(args,data, encoder, decoder):
    #input_sentence, kb, target_sentence = next(data.random_batch(1))

    if args.test:
        batch=data.getTestingBatch(1)[0]
    else:
        batch = data.getBatches(1, valid=True, transpose=False)[0]

    input_sentence = batch.encoderSeqs
    target_sentence = batch.decoderSeqs
    evaluate_and_show_attention(args,data,encoder, decoder, input_sentence, target_sentence, batch.encoderSeqsLen,batch.decoderSeqsLen)


def evaluate_model(args, data, encoder, decoder, valid=False):

    if args.test:
        batches = data.getTestingBatch(args.batch_size)
    elif valid:
        batches = data.getBatches(args.batch_size, valid=True, transpose=False)
    else:
        batches = data.getBatches(args.batch_size, test=True, transpose=False)


    all_predicted = []
    target_batches = []
    individual_metric = []

    for batch in batches:
        input_seq = batch.encoderSeqs
        target_seq = batch.decoderSeqs

        batch_prediction, batch_bleu = evaluate_and_calculate_blue(args, data, encoder, decoder, input_seq,
                                                                  target_seq,
                                                                  batch.encoderSeqsLen, batch.decoderSeqsLen)
        all_predicted.append(batch_prediction)
        target_batches.append(target_seq)
        individual_metric.append(batch_bleu)

    candidates, references = data.get_candidates(target_batches, all_predicted)

    global_metric_score = nltk.translate.bleu_score.corpus_bleu(references, candidates)

    candidates2, references2 = data.get_candidates(target_batches, all_predicted, True)

    moses_multi_bleu_score= moses_multi_bleu(references2, candidates2, True)

    return global_metric_score, individual_metric, moses_multi_bleu_score


def evaluate_and_calculate_blue(args, data, encoder, decoder, input_seq, target_seq, input_length, output_length):

    batch_predictions,intents = evaluate(args, encoder, decoder, input_seq, target_seq, input_length, output_length)
    batch_predictions = batch_predictions.transpose(0, 1)
    batch_metric_score = 0

    for i, sen in enumerate(batch_predictions):
        predicted = data.sequence2str(sen.numpy(), clean=True)
        reference = data.sequence2str(target_seq[i], clean=True)
        batch_metric_score += nltk.translate.bleu_score.sentence_bleu([reference], predicted)

    batch_metric_score= batch_metric_score/len(target_seq)

    return batch_predictions, batch_metric_score


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


def evaluate_and_show_attention(args, vocab, encoder, decoder, input_sentence, target_sentence, input_length, output_length):
    output_words, attentions = evaluate_sample(args,vocab, encoder, decoder, input_sentence, input_length, output_length)
    print(output_words)
    output_sentence = ' '.join(output_words)
    input_sentence= vocab.sequence2str(input_sentence[0],clean=True)
    print('>', input_sentence)
    if target_sentence is not None:
        target_sentence= vocab.sequence2str(target_sentence[0])
        print('=', target_sentence)
    print('<', output_sentence)

    #show_attention(input_sentence, output_words, attentions)

    # Show input, target, output text in visdom
    win = 'evaluted (%s)' % hostname
    text = '<p>&gt; %s</p><p>= %s</p><p>&lt; %s</p>' % (input_sentence, target_sentence, output_sentence)
    # vis.text(text, win=win, opts={'title': win})
