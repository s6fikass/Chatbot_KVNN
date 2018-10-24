# get data
from corpus.textdata import TextData
from model.vanilla_seq2seq import Seq2SeqmitAttn
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

train_file = 'data/kvret_train_public.json'
valid_file = 'data/kvret_dev_public.json'
test_file = 'data/kvret_test_public.json'
textdata = TextData(train_file, valid_file, test_file)

print('Datasets Loaded.')
print('Compiling Model.')

# Configure models
attn_model = 'dot'
n_layers = 1
dropout = 0.1
hidden_size = 200

# Configure training/optimization
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_epochs = 1000
epoch = 0
plot_every = 20
evaluate_every = 20
batch_size=10

Vanilla_model = Seq2SeqmitAttn(hidden_size, textdata.getTargetMaxLength(), textdata.getVocabularySize(), batch_size, hidden_size, textdata.word2id['<go>'], textdata.word2id['<eos>'], None, gpu=False, lr=0.01, train_emb=False,
                 n_layers=1, clip=2.0, pretrained_emb=None, dropout=0.1, emb_drop=0.2, teacher_forcing_ratio=0.0)

while epoch < n_epochs:
    epoch += 1
    # steps_done = 0
    batches = textdata.getBatches(batch_size, transpose=False)

    # steps_per_epoch = len(batches)
    try:
        epoch_loss = 0
        epoch_ec = 0
        epoch_dc = 0

        # while steps_done < steps_per_epoch:
        for current_batch in tqdm(batches, desc='Processing batches'):
            # current_batch=batches[steps_done]
            x = current_batch.encoderSeqs
            y = current_batch.targetSeqs

            xx = current_batch.encoderMaskSeqs
            yy = current_batch.decoderMaskSeqs
            print (len(xx))

            input_batch = Variable(torch.LongTensor(x)).transpose(0, 1)
            print(input_batch.size())

            target_batch = Variable(torch.LongTensor(y)).transpose(0, 1)
            input_batch_mask = Variable(torch.FloatTensor(xx)).transpose(0, 1)
            print(input_batch_mask)
            target_batch_mask = Variable(torch.FloatTensor(yy)).transpose(0, 1)
            Vanilla_model.train_batch(input_batch,target_batch,input_batch_mask,target_batch_mask)
        print("epoch-loss",Vanilla_model.loss/len(batches))
    except KeyboardInterrupt as e:
        print('Model training stopped early.')
        break

