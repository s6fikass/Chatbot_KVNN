from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from tensorflow.python.platform import gfile
import tensorflow as tf
import json

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 10000 == 0:
          print("  processing line %d" % counter)
        line = tf.compat.as_bytes(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")

def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True):
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 10000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                            tokenizer, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

# Extract Input and Target sentences (In Our dataSet input is from driver and output from the system)
def jsonPreProcess(self, filename, Mode):

    # create
    if not gfile.Exists(filename+Mode+'_input.txt'):
        lineID = 0
        conversationId = 0
        kb = {}
        datastore = ''
        with open(filename+'.json', 'r') as f:
            datastore = json.load(f)

        # Use the new datastore datastructure
        for dialogue in datastore:
            conversationId = conversationId+1
            conversationIds = []
            for utterence in dialogue["dialogue"]:
                if utterence['turn'] == 'driver':
                    with open(filename+Mode+'_input.txt', 'a') as f:
                        lineID = lineID+1
                        conversationIds.append("'L{0}'".format(lineID))
                        f.write('L{0}'.format(lineID)+' +++$+++ '+ utterence['data']['utterance'] + '\n')
                        f.close()
                else:
                    with open(filename+Mode+'_target.txt', 'a') as f:
                        lineID = lineID + 1
                        conversationIds.append("'L{0}'".format(lineID))
                        f.write('L{0}'.format(lineID) + ' +++$+++ ' + utterence['data']['utterance'] + '\n')
                        f.close()
            with open('{0}_conversations.txt'.format(filename), 'a') as c:
                c.write('C{0}'.format(conversationId)+' +++$+++ '+'['+', '.join(conversationIds)+']')
                c.write('\n')
                c.close()
            kb['C{0}'.format(conversationId)] = dialogue['scenario']['kb']['items']
        with open('{0}kb_items.json'.format(filename), 'a') as kbfile:
            json.dump(kb, kbfile)
        return [filename+Mode+'_input.txt', filename+Mode+'_target.txt', '{0}_conversations.txt'.format(filename),
                '{0}kb_items.json'.format(filename)]


def prepare_data(data_dir, vocabulary_size, tokenizer=None):
	# Get data to the specified directory.
	train_path='data/kvret_train_public'
	dev_path='data/kvret_dev_public'



	in_data_path='data/train_sof_in.csv'
	out_data_path='data/train_sof_out.csv'

	in_valid_path="data/valid_sof_in.csv"
	out_valid_path="data/valid_sof_out.csv"

	# Create vocabularies of the appropriate sizes.
	vocab_path = os.path.join(data_dir, "vocab%d" % vocabulary_size)
	create_vocabulary(vocab_path, in_data_path, vocabulary_size, tokenizer)

	# Create token ids for the training data.
	# in_train_ids_path = train_path + (".ids%d.in" % vocabulary_size)
	# out_train_ids_path = train_path + (".ids%d.out" % vocabulary_size)
	#
	# data_to_token_ids(in_data_path, in_train_ids_path, vocab_path, tokenizer)
	# data_to_token_ids(out_data_path, out_train_ids_path, vocab_path, tokenizer)
    #
	# # Create token ids for the development data.
	# in_dev_ids_path = dev_path + (".ids%d.in" % vocabulary_size)
	# out_dev_ids_path = dev_path + (".ids%d.out" % vocabulary_size)
	# data_to_token_ids(in_valid_path, in_dev_ids_path, vocab_path, tokenizer)
	# data_to_token_ids(out_valid_path, out_dev_ids_path, vocab_path, tokenizer)

	return (out_train_ids_path, in_train_ids_path,
          out_dev_ids_path, in_dev_ids_path,
          vocab_path)






