#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import string
import numpy as np
import random
import csv
import re
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model


# ### Reading 40k samples

# In[2]:


file = open('./sample_40K.txt','rt')
data = file.readlines()
file.close()


# In[3]:


def cleaning_data(text):
    
    text = text.lower()
    
    text = text.encode('ascii', 'ignore').decode()
    text = re.sub("\'\w+", '', text)
    
    # Remove punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    
    text = re.sub('\s{2,}', " ", text)
    
    text = text.split()
    return text


# In[4]:


def load_40K_clean_reviews(data):
    clean_data = []
    for i in range(len(data)):
        clean_data.append(cleaning_data(data[i]))
    return clean_data


# In[5]:


def replace_unk(data):
    freq = {}
    for sentence in data:
        for word in sentence:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1
    for i in range(len(data)):
        for j in range(len(data[i])):
            if freq[data[i][j]] <3 :
                data[i][j] = "unk"
    return data


# In[6]:


def corpus_to_vocab(corpus):
    
    word_to_ix, ix_to_word = {}, {}
    ix = 0
    
    for document in corpus:
        #print(document)
        for word in document:
            #print(word)
            #break
            if word not in word_to_ix.keys():
                word_to_ix[word], ix_to_word[ix] = ix, word
                ix += 1
            #break    
    return word_to_ix, ix_to_word


# In[7]:


clean_data = load_40K_clean_reviews(data)


# In[8]:


clean_data = replace_unk(clean_data)


# In[9]:


word_to_ix, ix_to_word = corpus_to_vocab(clean_data)


# In[10]:


N_WORDS = len(word_to_ix.keys())
print(N_WORDS)


# In[ ]:





# In[11]:


wordFreq = defaultdict(int)

for document in clean_data:
    for word in document:
        wordFreq[word] += 1


# In[12]:


totalWords = sum([freq for freq in wordFreq.values()])
wordProb = {word:(freq/totalWords) for word, freq in wordFreq.items()}


# In[13]:


a = sum([prob for prob in wordProb.values()])


# In[14]:


a


# In[16]:


def generate_negative_sample(wordProb):
      
    neg_samples  = list((np.random.choice(list(wordProb.keys()), p=list(wordProb.values())) for _ in range(5)))
    return neg_samples


# In[17]:


samples = generate_negative_sample(wordProb)


# In[18]:


samples


# ### Adding positive and negative samples

# In[ ]:





# In[19]:


posTrainSet =[]
negTrainSet = []
X=[]
y=[]


# In[ ]:


# add positive and negative examples
count = 0
for document in clean_data:
    for i in range(2, len(document)-2):
        #A = []
        word = word_to_ix[document[i]]
        #print(word)
        context_words = [word_to_ix[document[i-2]], word_to_ix[document[i-1]], word_to_ix[document[i+1]],word_to_ix[document[i+2]]]
        #print(context_words)
        posTrainSet.append((context_words[0],context_words[1], context_words[2], context_words[3], word))
        #A = list(range(len(document)))
        #A.remove(i-1)
        #A.remove(i)
        #A.remove(i+1)
        #print (A)
        #print ("XXXXXX")
        sample = generate_negative_sample(wordProb)
        neg_samples = [word_to_ix[sample[0]], word_to_ix[sample[1]], word_to_ix[sample[2]],word_to_ix[sample[3]],word_to_ix[sample[4]]]
        negTrainSet.append((neg_samples[0],neg_samples[1], neg_samples[2], neg_samples[3], neg_samples[4]))
        #break
        count +=1
    print (count)
n_pos_examples = len(posTrainSet)
n_neg_examples = len(negTrainSet)


# In[ ]:


X = np.concatenate([np.array(posTrainSet), np.array(negTrainSet)], axis=0)
y = np.concatenate([[1]*n_pos_examples, [0]*n_neg_examples])


# In[ ]:


embedding_layer = layers.Embedding(N_WORDS, EMBEDDING_DIM, 
                                   embeddings_initializer="RandomNormal",
                                   input_shape=(5,))


# In[ ]:


model = keras.Sequential()
model.add(embedding_layer)
model.add(layers.GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(X,y,epochs=10,shuffle=True)


# In[ ]:


model.save("my_model")


# In[ ]:


embeddings = embedding_layer.get_weights()[0]


# ### Saving files

# In[ ]:


with open("embedding_cbow", "wb") as f:
    pickle.dump(embeddings,f)

with open("index_to_word", "wb") as f:
    pickle.dump(ix_to_word,f)

with open("word_to_index", "wb") as f:
    pickle.dump(word_to_ix,f)	


# In[ ]:





# ### Reading saved files

# In[ ]:


with open("embedding_cbow", "rb") as f:
    Emb = pickle.load(f)
    
with open("word_to_index", "rb") as f:
    w_i = pickle.load(f)

with open("index_to_word", "rb") as f:
    i_w = pickle.load(f)


# ### Enter word

# In[ ]:


word = 'devices'


# In[ ]:





# In[ ]:


def find_top_n_similar(word, Emb, n=10):
    id_ = w_i[word]
    vec_word = Emb[id_, :]
    norm_vec_word = np.linalg.norm(vec_word)
    cos_sim = np.dot(Emb, vec_word.T) / (np.linalg.norm(Emb, axis=1) * norm_vec_word)
    top_n_ind = np.argsort(cos_sim)[-n:][::-1]
    return [i_w[id_] for id_ in top_n_ind]


# In[ ]:





# In[ ]:


reconstructed_model = keras.models.load_model("my_model")


# In[ ]:


top_words = find_top_n_similar(word, Emb, n=10)


# In[ ]:


print(top_words)


# In[ ]:


k =[w_i[i] for i in top_words]


# In[ ]:





# In[ ]:


tsne = TSNE()
p = [Emb[i,:] for i in k]
embed_tsne = tsne.fit_transform(p)


# In[ ]:




# ### Pre-trained Word2Vec

# In[ ]:


from gensim.models import Word2Vec, KeyedVectors


# In[ ]:


model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary = True)


# In[ ]:


model.wv.most_similar('titanic')

