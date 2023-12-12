from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def preprocess_text(text):
    return word_tokenize(text.lower())

def train_word2vec_model(inputs):
    preprocessed_inputs = [preprocess_text(input_text) for input_text in inputs]
    model = Word2Vec(sentences=preprocessed_inputs, vector_size=100, window=5, min_count=1, workers=4)
    return model

def calculate_semantic_similarity(model, input1, input2):
    tokens1 = preprocess_text(input1)
    tokens2 = preprocess_text(input2)
    
    vector1 = np.mean([model.wv[word] for word in tokens1 if word in model.wv], axis=0)
    vector2 = np.mean([model.wv[word] for word in tokens2 if word in model.wv], axis=0)
    
    return cosine_similarity([vector1], [vector2])[0][0]

def sort_inputs(inputs):
    word2vec_model = train_word2vec_model(inputs)

    sorted_indices = [0]
    remaining_indices = list(range(1, len(inputs)))

    while remaining_indices:
        last_index = sorted_indices[-1]
        best_index = min(remaining_indices, key=lambda x: calculate_semantic_similarity(word2vec_model, inputs[last_index], inputs[x]))
        sorted_indices.append(best_index)
        remaining_indices.remove(best_index)

    sorted_inputs = [inputs[i] for i in sorted_indices]

    return sorted_inputs



#TESTING

from test_sets import user_input_1
from test_sets import user_input_2
from test_sets import user_input_3
from test_sets import user_input_4

user_inputs = []

#user_inputs.extend(user_input_1)
#user_inputs.extend(user_input_2)
#user_inputs.extend(user_input_3)
#user_inputs.extend(user_input_4)   

sorted_inputs = sort_inputs(user_inputs)

print("Original Inputs:")
print(user_inputs)
print("\nSorted Inputs:")
print(sorted_inputs)
