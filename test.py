
#
from torch.autograd import Variable

from corpus.textdata import TextData
import torch
import torch
#
glove_filename = "data/samples/joint_model"

textdata = TextData("data/kvret_train_public.json", "data/kvret_dev_public.json",
                    "data/kvret_test_public.json")

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# x = textdata.word_to_embedding_dict["parking_garage"].reshape(1, -1)
# y = textdata.word_to_embedding_dict["dish_parking"].reshape(1, -1)
# y2 = textdata.word_to_embedding_dict["clear_sky"].reshape(1, -1)


# print(cosine_similarity(x,y2))
batches = textdata.getBatches(120, valid=True, transpose=False)

for batch in batches:
    input_batch = Variable(torch.LongTensor(batch.encoderSeqs)).transpose(0, 1)
    for i in batch.encoderMaskSeqs:
        print(len(i))
    target_batch = Variable(torch.LongTensor(batch.targetSeqs)).transpose(0, 1)

    input_batch_mask = Variable(torch.FloatTensor(batch.encoderMaskSeqs)).transpose(0, 1)
    target_batch_mask = Variable(torch.FloatTensor(batch.decoderMaskSeqs)).transpose(0, 1)