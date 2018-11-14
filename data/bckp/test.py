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
# # BLEUscore = nltk.
# +
# translate.bleu_score.corpus_bleu([reference,reference], hypothesis)
# # print BLEUscore
# import re
# print(' '.join(re.split('(\d+)(?=[a-z]|\-)', "the nearest parking garage is dish parking at 550 alester ave.,kjh 2pm")).strip())
# s='the nearest _entity_1_ is dish parking at 550 alester ave. would you like directions there?'
# k='550 alester ave'
# #
import numpy as np
# [0,0,0,0,0]  + [self.padToken] * (self.maxLengthDeco - len(batch.targetSeqs[i]))
# print(list(np.ones(5)))
# # print(np.ones(5))
x="-0.021613 -0.010357781 -0.009289622 -0.015169467 0.007073922 -0.023636546 -0.022265263 0.011377801 0.0059462935 -0.0034944469 0.009088421 0.012740096 -0.018048033 -0.0091935145 -0.0049544433 0.010829066 0.005335045 0.0011023758 -0.008142981 0.013967732 -0.012222893 -0.0054211398 0.0051305834 -0.018101173 0.009544491 -0.015290296 0.00048421812 -0.008943021 0.01167996 -0.0028148023 0.008296398 -0.008456702 0.0039602146 0.0031064488 -0.00041026436 -0.013065387 0.006534906 -0.007193583 -0.012207671 -0.0014755218 -0.0065317345 -0.01471643 -0.020478558 -0.010500515 0.002982656 -0.00899892 -0.003114838 -0.015142449 0.001866037 0.0031620096 0.01854237 -0.00057387794 -0.008095257 -0.023450585 0.017632566 -0.01228565 0.0052046683 0.012301761 0.0006115164 -0.0035992786 -0.011817155 -0.0030481382 -0.016149607 -0.028149622 -0.004373609 0.009945015 0.00026304275 -0.010422721 0.03729265 0.007362431 0.004732893 -0.02158914 0.0328231 -0.0049138255 0.0037536179 0.009068046 0.006199397 -0.0028205991 -0.0033241438 -0.0054002004 -0.001766121 -0.019000828 0.0060709594 0.0014368752 0.006420698 0.015416358 -0.015659828 0.012280254 -0.00091704354 -0.007679492 -0.009298168 -0.008400357 -0.001043896 -0.010114859 -0.022615248 0.00944721 -0.012287129 -0.037782386 -0.012261044 3.4914483e-06 0.00823467 0.016104322 0.021755647 -0.0034110388 0.017667372 -0.008234853 0.016764354 0.009274146 -0.005323629 -0.009347548 0.019257471 0.01863777 0.007761592 0.0039914763 -0.013891202 -0.014866219 -0.00438161 0.008669145 -0.013032436 0.015924137 0.0005877279 -0.0297161 -0.00861328 0.010776833 -0.009930693 -0.0072982213 -0.005982357 -0.014748413 0.0070922105 -0.008324545 0.010363629 0.013380101 0.0104574645 0.011378732 0.007474107 -0.01940263 0.0025900456 -0.0115682585 0.013722919 0.0016158769 -0.003081437 0.009671761 0.023976821 -0.0039490694 -0.0076169614 0.0032466191 0.009519948 -0.0016542602 -0.00048205536 -0.0014399268 0.009155986 0.004267307 0.009437639 0.0039261957 -0.01767031 -0.012992306 0.0010165492 -0.0073981034 0.0034511331 0.001773128 0.0010783365 -0.002043485 0.016933966 -0.019866802 0.010150153 0.0011045637 0.00079177786 0.007554171 -0.013869334 0.00022101263 0.021210376 -0.0073720627 0.006456394 0.016302947 -0.00579158 -0.021843264 -0.0048140343 0.0032605268 -0.023891529 -0.010276959 -0.01969607 0.020597288 -0.015216652 -0.006157998 -0.011129701 -0.0011090494 0.0035743467 0.01383896 -0.022558719 0.026812347 -0.0102957655 0.0033667078 -0.021702195 -0.020207064 -0.02147769 0.008356291 0.02183979 -0.007782441 0.005639326 -0.022694068"
x = x.split(" ")
sum=0
for i in range(len(x)):
    sum += float(x[i])
print(sum)

from corpus.textdata import TextData
#
#textdata = TextData("data/kvret_train_public.json","data/kvret_dev_public.json","data/kvret_test_public.json", "data/samples/jointEmbedding.txt")

batches = textdata.getBatches(1,transpose=False)
print(batches[0].targetKbMask)
print(batches[0].targetSeqs)
print(textdata.sequence2str(batches[0].targetSeqs[0]))

# batches = textdata.getBatches(1, valid=True, transpose=False)
# print(textdata.sequence2str(batches[0].decoderSeqs[0]))
import numpy as np
# import re
# x=re.split(',' ,'p.f. changs'.lower())
# for k in x:
#     xz=" ".join(re.split('^.|(\d+)(?=[a-z]|\-)', k.strip()))
#     zz=re.findall(r"[\w']+|[^\s\w']", xz)
#     print("_".join(zz))
import csv

# x.append(5)
# a=x.pop()
# print(x)
# x.append(a)
# print(x)
texts=[]
# b=textdata.getBatches(1,transpose=False)[0]
# print(b.targetSeqs)
# print(textdata.sequence2str(b.targetSeqs[0]))
# print(b.decoderMaskSeqs[0])
#
# for i in textdata.trainingSamples:
#     texts.append(textdata.sequence2str(i[0]).split())
#     texts.append(textdata.sequence2str(i[1]).split())
#
# # for i in textdata.validationSamples:
# #     texts.append(textdata.sequence2str(i[0]).split())
# #     texts.append(textdata.sequence2str(i[1]).split())
# #
# # for i in textdata.testSamples:
# #     texts.append(textdata.sequence2str(i[0]).split())
# #     texts.append(textdata.sequence2str(i[1]).split())
# #
# with open("data/samples/emb_in.txt", "w") as output:
#     writer = csv.writer(output, lineterminator='\n', delimiter =' ',quotechar =',',quoting=csv.QUOTE_MINIMAL)
#     writer.writerows(texts)
#
# # print (textdata.sequence2str(textdata.trainingSamples[0][0]))
# # print (textdata.sequence2str(textdata.trainingSamples[0][1]))
# # print (textdata.trainingSamples[0][2])
# # print (textdata.getMaxTriples())
# # # import numpy as np
# # # x=np.array2string(np.array(range(4)))
# # # print(x.strip("[").strip("]"))
# input_txt_conversation=[];
# input_txt_conversation.append("hi")
# input_txt_conversation.append("end")
# print(input_txt_conversation)
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
#(textdata.word_to_embedding_dict["cat"].reshape(1, -1),textdata.word_to_embedding_dict["cat"].reshape(1, -1))[0,0])