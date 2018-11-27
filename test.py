
#
from corpus.textdata import TextData
#
glove_filename = "data/samples/joint_model"

textdata = TextData("data/kvret_train_public.json", "data/kvret_dev_public.json",
                    "data/kvret_test_public.json", glove_filename)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
x = textdata.word_to_embedding_dict["parking_garage"].reshape(1, -1)
y = textdata.word_to_embedding_dict["dish_parking"].reshape(1, -1)
y2 = textdata.word_to_embedding_dict["clear_sky"].reshape(1, -1)


print(cosine_similarity(x,y2))