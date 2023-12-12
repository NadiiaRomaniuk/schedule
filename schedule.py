from gensim.models import Word2Vec as modl
from nltk.tokenize import word_tokenize as tok
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def prep_text(text):
    return tok(text.lower())  #конвертація у нижній регістр + токенізація слів

def train(inputs):   
    prep_inp = [prep_text(input_text) for input_text in inputs]
    model = modl(sentences=prep_inp, vector_size=100, window=5, min_count=1, workers=8) #workers = 8)
    return model

def similarity(model, input1, input2):
    tokens1 = prep_text(input1)
    tokens2 = prep_text(input2)
    
    vector1 = np.mean([model.wv[word] for word in tokens1 if word in model.wv], axis=0)
    vector2 = np.mean([model.wv[word] for word in tokens2 if word in model.wv], axis=0)
    
    return cosine_similarity([vector1], [vector2])[0][0]

def sort_inputs(inputs):
    word2vec_modl = train(inputs)

    sorted = [0]
    remaining = list(range(1, len(inputs)))
    count = 0

    while remaining:
        last_index = sorted[-1]
        best_index = min(remaining, key=lambda x: similarity(word2vec_modl, inputs[last_index], inputs[x]))   #вибирається індекс, що є найменш подібним за значенням до попереднього, вже внесеного у відсортований список
        sorted.append(best_index)
        remaining.remove(best_index)
        count = count + 1
    sorted_inputs = [inputs[i] for i in sorted]

    return sorted_inputs



#TESTING

from test_sets import user_input_1
from test_sets import user_input_2
from test_sets import user_input_3
from test_sets import user_input_4
from test_sets import user_input_5

user_inputs = []
rest=['rest', 'rest', 'rest', 'rest']

user_inputs.extend(user_input_1) 
user_inputs.extend(rest)

#user_inputs.extend(user_input_2)
#user_inputs.extend(rest)

#user_inputs.extend(user_input_3)
#user_inputs.extend(rest)

#user_inputs.extend(user_input_4) 
#user_inputs.extend(rest)

#user_inputs.extend(user_input_5)      #oтут банальний приклад, але наочний 
#user_inputs.extend(rest)

sorted_inputs = sort_inputs(user_inputs)

print("Original Inputs:")
print(user_inputs)
print("\nSorted Inputs:")
print(sorted_inputs)
