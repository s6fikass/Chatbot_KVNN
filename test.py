# import re
#
# line ="You're welcome!?"
# x=re.split('[!?.,]',line)
# sentencesToken = re.findall(r"[\w']+|[^\s\w']", line)
# print x
# print sentencesToken

# import nltk
#
# hypothesis = [['It', 'is', 'a', 'cat', 'at', 'room'], ['It', 'is', 'a', 'cat', 'at', 'room']]
# reference = [['It', 'is', 'a', 'cat', 'inside', 'the', 'room']]
# #there may be several references
# # BLEUscore = nltk.translate.bleu_score.corpus_bleu([reference,reference], hypothesis)
# # print BLEUscore
# from corpus.textdata import TextData
# #

# print (textdata.sequence2str(textdata.trainingSamples[0][0]))
# print (textdata.sequence2str(textdata.trainingSamples[0][1]))
# print (textdata.trainingSamples[0][2])
# print (textdata.getMaxTriples())

# batches=textdata.getBatches(500)
# print batches[0].kb_inputs
# print len(batches[0].kb_inputs)
# # import json
# #
# with open('kb.json', 'r') as f:  # TODO: Solve Iso encoding pb !
#     datastore = json.load(f)
#     for kb in datastore:
#         for triple in kb['kb']['items']:
#             print triple
# for input_batch, target_batch, batch_preds in zip(input_batches, target_batches, all_preds):
                    #     for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
                    #         print('\t', input_sent)
                    #         print('\t => ', idx2sent(pred, reverse_vocab=dec_reverse_vocab))
                    #         print('\tCorrent answer:', target_sent)
# a = [1,2,3,4,5,6]
# b = a[:5]
#
# print(b)
# import re
# l=re.findall(r"[\w']+|[^\s\w']", "where's the nearest <entity_1>")
# # x="<entity_1>?".strip(",").strip(".").strip(":").strip("?").\
#     #                strip("!").strip(";").strip(' \n\t').strip()
# print(l)
# import  numpy as np
# from collections import defaultdict
#
# GLOVE_FILENAME = "data/glove_data/glove.840B.300d.txt"
# def load_embedding_from_disks(glove_filename, with_indexes = False):
#     """
#     Read a GloVe txt file. If `with_indexes=True`, we return a tuple of two dictionnaries
#     `(word_to_index_dict, index_to_embedding_array)`, otherwise we return only a direct
#     `word_to_embedding_dict` dictionnary mapping from a string to a numpy array.
#     """
#     if with_indexes:
#         word_to_index_dict = dict()
#         index_to_embedding_array = []
#     else:
#         word_to_embedding_dict = dict()
#
#     with open(glove_filename, 'r') as glove_file:
#         for (i, line) in enumerate(glove_file):
#
#             split = line.split(' ')
#
#             word = split[0]
#
#             representation = split[1:]
#             representation = np.array(
#                 [float(val) for val in representation]
#             )
#
#             if with_indexes:
#                 word_to_index_dict[word] = i
#                 index_to_embedding_array.append(representation)
#             else:
#                 word_to_embedding_dict[word] = representation
#
#     _WORD_NOT_FOUND = [0.0] * len(representation)  # Empty representation for unknown words.
#     if with_indexes:
#         _LAST_INDEX = i + 1
#         word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
#         index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
#         return word_to_index_dict, index_to_embedding_array
#     else:
#         word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
#         return word_to_embedding_dict
#
# print("Loading embedding from disks...")
# word_to_embedding_dict = load_embedding_from_disks(GLOVE_FILENAME, with_indexes=True)
# print("Embedding loaded from disks.")
from sklearn.feature_extraction.text import TfidfVectorizer

documents = (
    "Macbook GPU",
    "Macbook GPU"
)

# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
from corpus.textdata import TextData

#textdata = TextData("data/kvret_train_public.json","data/kvret_dev_public.json","data/kvret_test_public.json")

# from sklearn.metrics.pairwise import cosine_similarity
# # print(textdata.word_to_embedding_dict["cat"])
# print(cosine_similarity(textdata.word_to_embedding_dict["cat"], textdata.word_to_embedding_dict["cat"]))


from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import torch
a=np.zeros(300)
b=np.zeros(300)
c=[12.3,23.1]
f=torch.Tensor(c)
f=f.sum()
f=f.add(9)
print(f.item())
v=np.add(a,b)
# print(v)
v=np.divide(v,2)

v=v.reshape(1, -1)
# a=np.pad(a, (0, 1), 'constant')
# print(a)
# a=a.reshape(1, -1)
#
# ab=ab.reshape(1, -1)
print(cosine_distances(v,v))
# print(cosine_distances(textdata.word_to_embedding_dict["cat"].reshape(1, -1),textdata.word_to_embedding_dict["cat"].reshape(1, -1))[0,0])