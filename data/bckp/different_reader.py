import json
import csv
import random
import operator

import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize


random.seed(1984)

INPUT_PADDING = 50
OUTPUT_PADDING = 100
PAD_token = 0
EOS_token = 1
SOS_token = 2
EOU_token = 3

class Vocabulary(object):

    def __init__(self, vocabulary_file, padding=None):
        """
            Creates a vocabulary from a file
            :param vocabulary_file: the path to the vocabulary
        """
        self.vocabulary_file = vocabulary_file
        with open(vocabulary_file, 'r',encoding='utf-8') as f:
            self.vocabulary = json.load(f)

        self.padding = padding
        self.reverse_vocabulary = {v: k for k, v in self.vocabulary.items()}

    def size(self):
        """
            Gets the size of the vocabulary
        """
        return len(self.vocabulary.keys())

    def string_to_int(self, text):
        """
            Converts a string into it's character integer 
            representation
            :param text: text to convert
        """
        tokens = text.split(" ")
        #print(tokens)
        integers = []

        if self.padding and len(tokens) >= self.padding:
            # truncate if too long
            tokens = tokens[-(self.padding - 1):]


        tokens.append('<eos>')

        for c in tokens:
            if c.strip(",").strip(".").strip(":").strip("?").strip("!").strip(";").strip(' \n\t') in self.vocabulary:
                integers.append(self.vocabulary[c.strip(",").strip(".").strip(":").strip("?")
                                .strip("!").strip(";").strip(' \n\t')])
            elif c.lower() in self.vocabulary:
                integers.append(self.vocabulary[c.lower()])
            else:
                integers.append(self.vocabulary['<unk>'])


        # pad:
        if self.padding and len(integers) < self.padding:
            integers.reverse()
            integers.extend([self.vocabulary['<pad>']]
                            * (self.padding - len(integers)))
            integers.reverse()

        if len(integers) != self.padding:
            print(text)
            raise AttributeError('Length of text was not padding.')
        return integers

    def int_to_string(self, integers):
        """
            Decodes a list of integers
            into it's string representation
        """
        tokens = []
        for i in integers:
            tokens.append(self.reverse_vocabulary[i])

        return tokens


class Data(object):

    def __init__(self, file_name, vocabulary,kb_vocabulary):
        """
            Creates an object that gets data from a file
            :param file_name: name of the file to read from
            :param vocabulary: the Vocabulary object to use
            :param batch_size: the number of datapoints to return
            :param padding: the amount of padding to apply to 
                            a short string
        """

        self.input_vocabulary = vocabulary
        self.output_vocabulary = vocabulary
        self.kb_vocabulary=kb_vocabulary
        self.kbfile = "./data/normalised_kbtuples.csv"
        self.file_name = file_name

    def kb_out(self):
        df=pd.read_csv(self.kbfile)

        self.kbs=list(df["subject"]+"_"+df["relation"])

        self.kbs = np.array(list(
            map(self.kb_vocabulary.string_to_int, self.kbs)))


    def load(self):
        """
            Loads data from a file
        """
        self.inputs = []
        self.targets = []
        self.trainingSamples = []

        with open(self.file_name, 'r',encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0]=="inputs":
                    continue
                #print(row[0],row[1])
                self.inputs.append(row[0])
                self.targets.append(row[1])

                if len(row) > 2:  # if Kb is included
                    self.trainingSamples.append([row[0], row[1], row[2]])
                else:
                    self.trainingSamples.append([row[0], row[1]])


    def transform(self):
        """
            Transforms the data as necessary
        """
        # @TODO: use `pool.map_async` here?
        self.inputs = np.array(list(
            map(self.input_vocabulary.string_to_int, self.inputs)))
        self.targets = np.array(list(map(self.output_vocabulary.string_to_int, self.targets)))


    def generator(self, batch_size):
        """
            Creates a generator that can be used in `model.fit_generator()`
            Batches are generated randomly.
            :param batch_size: the number of instances to include per batch
        """
        instance_id = range(len(self.inputs))
        while True:
            try:
                batch_ids = random.sample(instance_id, batch_size)
                targets=np.array(self.targets[batch_ids])
                targets = np.array(list(map(lambda x: to_categorical(x,num_classes=self.output_vocabulary.size()),targets)))

                yield ([np.array(self.inputs[batch_ids], dtype=int), np.repeat(self.kbs[np.newaxis,:,:],batch_size,axis=0)], np.array(targets))
            except Exception as e:
                print('EXCEPTION OMG')
                print(e)
                yield None, None,None

    def batchGenerator(self, batch_size):
        instance_id = range(len(self.inputs))
        while True:
            try:
                batch_ids = random.sample(instance_id, batch_size)
                targets = np.array(self.targets[batch_ids])

                #targets = np.array(list(map(lambda x: to_categorical(x,num_classes=self.output_vocabulary.size()),targets)))
                yield (np.array(self.inputs[batch_ids], dtype=int),
                        np.repeat(self.kbs[np.newaxis, :, :], batch_size, axis=0), np.array(targets))
            except Exception as e:
                print('EXCEPTION OMG')
                print(e)
                yield None, None, None

    def random_batch(self, batch_size):
        instance_id = range(len(self.inputs))
        # instance_id=range(batch_size)

        while True:
            try:
                batch_ids = random.sample(instance_id, batch_size)
                targets = np.array(self.targets[batch_ids])
                inputs = np.array(self.inputs[batch_ids], dtype=int)
                kbs = np.array(self.inputs[batch_ids], dtype=int)

                #targets = np.array(list(map(lambda x: to_categorical(x,num_classes=self.output_vocabulary.size()),targets)))
                yield (inputs,
                        np.repeat(self.kbs[np.newaxis, :, :], batch_size, axis=0), targets)
            except Exception as e:
                print('EXCEPTION OMG')
                print(e)
                yield None, None, None


if __name__ == '__main__':

    vocab = Vocabulary('./data/vocabulary.json', padding=20)
    kb_vocabulary = Vocabulary('./data/vocabulary.json', padding=4)
    print(vocab.string_to_int("find the address to a hospital or clinic. hospital#poi is at Stanford_Express_Care#address. thank you."))
    ds = Data('./data/train_data.csv', vocab, kb_vocabulary)
    ds.kb_out()
    g = ds.generator(1)
    ds.load()
    ds.transform()
    x,r,y=next(ds.random_batch(2))

    #print(vocab.string_to_int("find starbucks <eos>"))
    for i in range(50):
         #print(next(g)[0][0][0], next(g)[1][0])

         x,y=next(g)
         print(vocab.int_to_string(list(x[0][0])))
         print(x[1][0].shape)
         # print(vocab.int_to_string(list(y[0])))
         # print(kb_vocabulary.int_to_string(x[1][0][2]))
         # print(vocab.int_to_string(list(x[0][0])))
         # print(vocab.int_to_string(list(next(g)[0][0][0])))



         break
