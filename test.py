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
#
# textdata=TextData("data/kvret_train_public.json","data/kvret_dev_public.json","data/kvret_test_public.json")
# print textdata.trainingSamples[0]
# print textdata.sequence2str(textdata.trainingSamples[0][0])
# print textdata.sequence2str(textdata.trainingSamples[0][1])
# print textdata.trainingSamples[0][2]
# print textdata.getMaxTriples()
#
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
a = [1,2,3,4,5,6]
b = a[:5]

print(b)