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
# textdata=TextData("data/kvret_train_public.json")
#
# import json
#
# with open('kb.json', 'r') as f:  # TODO: Solve Iso encoding pb !
#     datastore = json.load(f)
#     for kb in datastore:
#         for triple in kb['kb']['items']:
#             print triple

l=[]
l2=[5, 6, 7, 8, 9]
l.extend(l2)
print l