#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
from sklearn.decomposition import TruncatedSVD


# In[ ]:





# In[2]:


# with open('sample_data.txt',w) as f:
#     for i in sample_data:
#         f.write(i + "\n")


# In[3]:


import json


with open('reviews_Movies_and_TV.json', 'r') as f:
    review_texts = []

    for i in range(40000):
        line = f.readline()

        data = json.loads(line)
        review_text = data['reviewText']
        review_texts.append(review_text)


# In[4]:


review_texts[39999]


# In[5]:


len(review_texts)


# In[6]:


import re

def clean_text(data):
    # Remove any HTML tags from the data
    data = re.sub('<[^>]*>', '', data)

    # Remove any URLs from the data
    data = re.sub('https?://\S+|www\.\S+', '', data)

    # Remove any non-alphanumeric characters from the data
    data = re.sub('[^a-zA-Z0-9\s]', '', data)

    # Convert the data to lowercase
    data = data.lower()

    # Remove any extra whitespaces from the data
    data = re.sub('\s+', ' ', data).strip()
    return data


# In[7]:


clean_data=[]
for i in review_texts:
    
    
    clean_data.append(clean_text(i))
    


# In[8]:


len(clean_data)


# In[9]:


clean_data[2]


# In[10]:


import nltk
from nltk.corpus import stopwords
#print(stopwords.words('english'))


# In[11]:


stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


# In[12]:


filter_data=[]
for sent in clean_data:
    temp=[]
    for word in sent.split():
        if word not in stopwords:
            temp.append(word)
    filter_data.append(temp)
# print(filter_data)
len(filter_data)


# In[13]:


total_words=[]
for i in filter_data:
    for j in i:
        total_words.append(i)
        


# In[14]:


len(total_words)


# In[15]:


a = list(set([word for line in filter_data for word in line]))


# In[16]:


len(a)


# In[ ]:





# In[17]:


len(filter_data)
filter_data[0]


# In[ ]:





# In[18]:


def replace_unk(data, min_freq=3, unk_token='unk'):
    freq = {}
    for sentence in data:
        for word in sentence:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1
    for i in range(len(data)):
        for j in range(len(data[i])):
            if freq[data[i][j]] < min_freq:
                data[i][j] = unk_token
    return data


# In[19]:


filter_data_1=replace_unk(filter_data,min_freq=3, unk_token='unk')
len(filter_data_1)


# In[20]:


all_words=[]
for sent in filter_data_1:
    all_words.extend(sent)
    
all_words= list(set(all_words))
all_words.sort()

# print(all_words)
print(len(all_words))


# In[ ]:





# In[21]:


# Create a dictionary of all unique words in the corpus and their corresponding indices
word2idx = {}
idx2word = {}
for i, word in enumerate(set([word for doc in filter_data_1 for word in doc])):
    word2idx[word] = i
    idx2word[i] = word


# In[22]:


len(word2idx)


# In[23]:


# Use SVD to factorize the co-occurrence matrix into word embeddings


# In[24]:


# Define the window size
window_size = 2

# Create a co-occurrence matrix
co_occurrence_matrix = np.zeros((len(word2idx), len(word2idx)))

for doc in filter_data_1:
    for i, word in enumerate(doc):
        for j in range(max(0, i - window_size), min(len(doc), i + window_size + 1)):
            if i != j:
                co_occurrence_matrix[word2idx[word]][word2idx[doc[j]]] += 1
                
                



# In[25]:


co_occurrence_matrix.shape


# In[26]:


from scipy.sparse import csr_matrix

co_occurrence_matrix_sparse = csr_matrix(co_occurrence_matrix)


# In[27]:


from scipy.sparse.linalg import svds

# Define the number of dimensions for the word embeddings
embedding_size = 200

# Use truncated SVD to factorize the sparse co-occurrence matrix into word embeddings
U, S, V = svds(co_occurrence_matrix_sparse, k=embedding_size)


# In[28]:


word_embeddings = U
len(word_embeddings[0])


# In[29]:


word_embeddings[2,:]


# In[ ]:





# In[ ]:





# In[30]:


def find_top_n_similar(word, embeddings, word2idx, n=10):
    # Get the vector representation of the word
    if word not in word2idx:
        return f"Error: '{word}' not found in vocabulary."
    word_idx = word2idx[word]
    word_vec = embeddings[word_idx]

    # Compute the cosine similarity between the word vector and all other word vectors
    similarities = np.dot(embeddings, word_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(word_vec))

    # Get the indices of the top n most similar words (excluding the input word itself)
    top_n_indices = np.argsort(similarities)[::-1][1:n+1]

    # Get the words corresponding to the top n indices
    top_n_words = [list(word2idx.keys())[idx] for idx in top_n_indices]

    return top_n_words


# In[31]:


word= "titanic"
similar_word=find_top_n_similar(word,word_embeddings,word2idx,n=10)


# In[32]:


similar_word


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[33]:





# In[34]:


from gensim.models import Word2Vec, KeyedVectors


# In[35]:


model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary = True)


# In[ ]:


model.wv.most_similar('titanic')

