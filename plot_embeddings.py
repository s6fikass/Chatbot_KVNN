import matplotlib.pyplot as plt
import hypertools as hyp
import numpy as np
from sklearn.decomposition import PCA
from corpus.textdata import TextData

glove_filename = "data/samples/jointEmbedding.txt"

textdata = TextData("data/kvret_train_public.json", "data/kvret_dev_public.json",
                    "data/kvret_test_public.json", glove_filename)

x = []
s=[]

plotting_sentences = ["will it be warm in camarillo over the next 2 days No it's not gonna be warm in Camarillo over the next 2 days", "show me directions to the nearest mall", "tom's house is 6_miles away at 580_van_ness_ave"]

# plt.scatter(representation[0], representation[1])
# plt.annotate(word, xy=(representation[0], representation[1]), xytext=(5, 2),
#              textcoords='offset points', ha='right', va='bottom')

for sentence in plotting_sentences:
    for word in sentence.split(" "):
        x.append(textdata.word_to_embedding_dict[word])
        s.append(word)

hyp.plot(np.array(x), '.', ndims=2, labels=s,
         reduce="IncrementalPCA", zoom=20, n_clusters=4)