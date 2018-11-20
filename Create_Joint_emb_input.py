from corpus.textdata import TextData
import csv


textdata = TextData("data/kvret_train_public.json","data/kvret_dev_public.json","data/kvret_test_public.json")

texts=[]

for i in textdata.trainingSamples:
    texts.append(textdata.sequence2str(i[0], clean=True).split())
    texts.append(textdata.sequence2str(i[1], clean=True).split())

with open("data/samples/emb_in.txt", "w") as output:
    writer = csv.writer(output, lineterminator='\n', delimiter =' ',quotechar =',',quoting=csv.QUOTE_MINIMAL)
    writer.writerows(texts)
