from __future__ import print_function
from corpus.textdata import TextData
from tf_model.seq2seq_model import Seq2Seq
#from tf_model.seq2seq_kv_model import Seq2SeqKV
import collections
import time
import math
import os
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops

import tensorflow as tf


def get_config_proto(log_device_placement=False, allow_soft_placement=True,
                     num_intra_threads=0, num_inter_threads=0):
  # GPU options:
  # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True

  # CPU threads options
  if num_intra_threads:
    config_proto.intra_op_parallelism_threads = num_intra_threads
  if num_inter_threads:
    config_proto.inter_op_parallelism_threads = num_inter_threads

  return config_proto


def get_candidates(target_batch, pridictions):
    candidate_sentences=[]
    reference_sentences=[]
    for target, pridiction in zip(target_batch, pridictions):
        reference_sentences.append([textData.sequence2str(target)])
        candidate_sentences.append(textData.sequence2str(pridiction))
    return candidate_sentences, reference_sentences

# input_batches = [
#     ['Hi What is your name?', 'Nice to meet you!'],
#     ['Which programming language do you use?', 'See you later.'],
#     ['Where do you live?', 'What is your major?'],
#     ['What do you want to drink?', 'What is your favorite beer?']]
#
# target_batches = [
#     ['Hi this is Jaemin.', 'Nice to meet you too!'],
#     ['I like Python.', 'Bye Bye.'],
#     ['I live in Seoul, South Korea.', 'I study industrial engineering.'],
#     ['Beer please!', 'Leffe brown!']]

print (tf.__version__)
attention = False
attention_architecture ="standard" #standard || KVAttention
train_file = 'data/kvret_train_public.json'
model_dir="trained_model"
textData = TextData(train_file)

voc_size = textData.getVocabularySize()
batch_size = 256
eos=1
maxLengthEnco = textData.getInputMaxLength()
maxLengthDeco = textData.getTargetMaxLength()

print ("max enc %d" %maxLengthEnco)

steps_per_eval = 100  # * steps_per_stats


train_graph = tf.Graph()
with train_graph.as_default(), tf.container("train"):

    model_device_fn = None
    with tf.device(model_device_fn):
        if attention_architecture == "standard":
            model = Seq2Seq(
                200,
                vocab_size=voc_size,
                encoder_len=maxLengthEnco,
                decoder_len=maxLengthDeco,
                batch_size=batch_size,
                stop_symbols=eos,
                use_attn=attention
            )
            if attention:
                model_dir = "trained_model/AttnSeq2Seq"
            else:
                model_dir = "trained_model/Seq2Seq"
        elif attention_architecture == "KVAttention":
            model_creator = Seq2SeqKV

    #train_sess = tf.Session()
    config_proto = get_config_proto(
      log_device_placement=False,
      num_intra_threads=0,
      num_inter_threads=0)

    train_sess = tf.Session(target="", config=config_proto, graph=train_graph)

"""Create translation model and initialize or load parameters in session."""
latest_ckpt = tf.train.latest_checkpoint(model_dir)
if latest_ckpt:
    model.saver.restore(train_sess, latest_ckpt)
    print("Model restored.")
    print("Current Global step", model.global_step.eval(train_sess))
    global_step = model.global_step.eval(train_sess)
else:
    start_time = time.time()
    with train_graph.as_default():
        train_sess.run(tf.global_variables_initializer())
    global_step = 0


n_epoch = 200

epoch_step = global_step
loss_history = []
while epoch_step < n_epoch:
    try:
        epoch_step += 1
        all_predicted = []
        epoch_loss = 0
        batches = textData.getBatches(batch_size)

        for current_step in range(0,len(batches)):
            nextBatch = batches[current_step]
            # Training pass
            feedDict = {}
            # print(textData.sequence2str(nextBatch.encoderSeqs[current_step]))
            # print(textData.sequence2str(nextBatch.decoderSeqs[current_step]))
            # print(nextBatch.weights[current_step])
            # print(textData.sequence2str(nextBatch.targetSeqs))
            # print (np.array(tf.unstack(nextBatch.encoderSeqs)).shape)

            model.update_feed_dict(feedDict, nextBatch.encoderSeqs,
                                                nextBatch.decoderSeqs, nextBatch.targetSeqs,
                                                nextBatch.weights)

            [out, batch_predictions, batch_loss, _] = train_sess.run([model.outputs, model.predictions, model.total_loss, model.training_op], feed_dict=feedDict)

            loss_history.append(batch_loss)
            epoch_loss += batch_loss
            all_predicted.append(batch_predictions)

            pred = []
            for out in out:
                pred.append(np.argmax(out))

            if epoch_step % 10 == 0:
                print('Epoch', epoch_step)
                #textData.printBatch(nextBatch)
                # for input_batch, target_batch, batch_preds in zip(input_batches, target_batches, all_preds):
                #     for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
                #         print('\t', input_sent)
                #         print('\t => ', idx2sent(pred, reverse_vocab=dec_reverse_vocab))
                #         print('\tCorrent answer:', target_sent)
                target_batch=np.transpose(nextBatch.decoderSeqs)
                candidates, references =get_candidates(target_batch,all_predicted[len(all_predicted)-1])
                BLEUscore = nltk.translate.bleu_score.corpus_bleu(references, candidates)
                print (BLEUscore)

                print ("=>",textData.sequence2str(all_predicted[len(all_predicted)-1][0]))
                #deco2sentence
                # print (pred[0])
                # print("=>", textData.sequence2str(pred))
                # print ('\tepoch loss: {:.2f}\n'.format(epoch_loss))

        print('Epoch', epoch_step)
        print('Training loss', loss_history[len(loss_history)-1])

    except (KeyboardInterrupt, SystemExit):  # If the user press Ctrl+C while testing progress
        print('Interruption detected, exiting the program...')
        # model.global_step = epoch_step
        # # Save checkpoint
        # model.saver.save(
        #     train_sess,
        #     os.path.join(model_dir, "translate.ckpt"),
        #     global_step=epoch_step)
        break

model.global_step=epoch_step
# Save checkpoint
model.saver.save(train_sess,
            os.path.join(model_dir, "translate.ckpt"),
            global_step=epoch_step)