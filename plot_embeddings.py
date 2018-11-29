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

plotting_sentences = ["i have a doctor appointment next month on the 13 th at 11 am with tom please set a reminder",
                      "navigate to my friend's home please do you know the address of friend's home actually i need to go home the quickest route please,the quickest route home is 4_miles away with heavy_traffic located at 5671_barringer_street",
                      "what's the forecast in carson for this weekend,in carson it will be foggy on saturday and dew on sunday"]

# plt.scatter(representation[0], representation[1])
# plt.annotate(word, xy=(representation[0], representation[1]), xytext=(5, 2),
#              textcoords='offset points', ha='right', va='bottom')

for sentence in plotting_sentences:
    for word in sentence.split(" "):
        x.append(textdata.word_to_embedding_dict[word])
        s.append(word)

hyp.plot(np.array(x), '.', ndims=2, labels=s,
         reduce="IncrementalPCA", zoom=20, n_clusters=4)