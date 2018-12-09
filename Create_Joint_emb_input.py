from corpus.textdata import TextData
import csv
from collections import Counter
import numpy as np
textdata = TextData("data/kvret_train_public.json","data/kvret_dev_public.json","data/kvret_test_public.json")

texts=[]

for i in textdata.trainingSamples:
    texts.append(textdata.sequence2str(i[0], clean=True).split(" "))
    texts.append(textdata.sequence2str(i[1], clean=True).split(" "))

with open("data/samples/emb_in.txt", "w") as output:
    writer = csv.writer(output, lineterminator='\n', delimiter =' ',quotechar =',',quoting=csv.QUOTE_MINIMAL)
    writer.writerows(texts)


comatrix = Counter()
for i in textdata.trainingSamples:
    for triple in i[2]:
        comatrix[(triple[0], triple[2])]+=1
kb=[]
kb_comb=[]
for (ei, ej) in comatrix:
    kb.append([textdata.id2word[ei],textdata.id2word[ej]])
ALPHA = 0.55
N=(comatrix.most_common(1)[0])[1]/100
for (ei, ej) in comatrix:
    x = min(((comatrix[(ei,ej)] **ALPHA)/N),1)
    kb_comb.append([textdata.id2word[ei],textdata.id2word[ej], str(comatrix[(ei,ej)])])


with open("data/samples/kb_emb_in.txt", "w") as output:
    writer = csv.writer(output, lineterminator='\n', delimiter =' ',quotechar =' ',quoting=csv.QUOTE_MINIMAL)
    writer.writerows(kb)

with open("data/samples/kb_emb_comb_in.txt", "w") as output:
    writer = csv.writer(output, lineterminator='\n', delimiter =' ',quotechar =' ',quoting=csv.QUOTE_MINIMAL)
    writer.writerows(kb_comb)

textdata = TextData("data/kvret_train_public.json","data/kvret_dev_public.json","data/kvret_test_public.json")

texts=[]

for i in textdata.testSamples:
    texts.append([textdata.sequence2str(i[0], clean=True),textdata.sequence2str(i[1], clean=True)])

with open("data/samples/test.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')#quotechar ='',quoting=csv.QUOTE_MINIMAL)
    writer.writerows(texts)
