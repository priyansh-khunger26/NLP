import conllu
from collections import Counter
import re
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


word_to_idx = {}
tag_to_idx = {}

with open('./UD_English-Atis/en_atis-ud-train.conllu', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
    for line in lines:
        if line.startswith('#') or line == '\n':
            continue
            
        tokens = line.strip().split('\t')
        
        # Get the word and tag from the CoNLL line
        word = tokens[1]
        tag = tokens[3]
        
        # Add the word to the word_to_idx dictionary if it does not already exist
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
        
        # Add the tag to the tag_to_idx dictionary if it does not already exist
        if tag not in tag_to_idx:
            tag_to_idx[tag] = len(tag_to_idx)


def read_conllu_file(file_path):    ## it return sentence
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                # End of sentence or comment line
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                fields = line.split('\t')
                if '-' in fields[0]:
                    # Multi-word token
                    continue
                word = fields[1]
                sentence.append(word)
        if sentence:
            sentences.append(sentence)
    return sentences



def read_conllu_file_1(file_path):    ## it return postag
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                # End of sentence or comment line
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                fields = line.split('\t')
                if '-' in fields[0]:
                    # Multi-word token
                    continue
                word = fields[1]
                pos = fields[3]
                sentence.append( pos)
        if sentence:
            sentences.append(sentence)
    return sentences



file_path='./UD_English-Atis/en_atis-ud-train.conllu'
sentence=read_conllu_file(file_path)
pos_tag=read_conllu_file_1(file_path)
dev_file_path='./UD_English-Atis/en_atis-ud-dev.conllu'
dev_sentence=read_conllu_file(dev_file_path)
dev_pos_tag=read_conllu_file_1(dev_file_path)
test_file_path='./UD_English-Atis/en_atis-ud-test.conllu'
test_sentence=read_conllu_file(test_file_path)
test_pos_tag=read_conllu_file_1(test_file_path)




# Create word-to-index dictionary
min_count=1
word_counts = Counter()
for sen in sentence:
    for word in sen:
        word_counts[word] += 1
vocab = {'<PAD>': 0, '<UNK>': 1}
for word, count in word_counts.items():
    if count >= min_count:
        vocab[word] = len(vocab)
word_to_idx = {word: idx for word, idx in vocab.items()}

# Convert sentences to tensors
import torch
sentences_tensors = []
for sen in sentence:
    # Convert words to indices using word-to-index dictionary
    word_indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in sen]
    # Convert indices to tensor
    sentence_tensor = torch.LongTensor(word_indices)
    sentences_tensors.append(sentence_tensor)
    
    
    

# Convert sentences to tensors
import torch
dev_sentences_tensors = []
for sen in dev_sentence:
    # Convert words to indices using word-to-index dictionary
    word_indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in sen]
    # Convert indices to tensor
    dev_sentence_tensor = torch.LongTensor(word_indices)
    dev_sentences_tensors.append(dev_sentence_tensor)
    # dev_sentences_tensors=torch.tensor(dev_sentences_tensors)
    
    


# Create tag-to-index dictionary
min_count=5
tag_counts = Counter()
for sen in pos_tag:
    for tag in sen:
        tag_counts[tag] += 1
vocab = {'<PAD>': 0, '<UNK>': 1}
for tag, count in tag_counts.items():
    if count >= min_count:
        vocab[tag] = len(vocab)
tag_to_idx = {word: idx for word, idx in vocab.items()}

# Convert sentences to tensors
import torch
tags_tensors = []
for sen in pos_tag:
    # Convert words to indices using word-to-index dictionary
    tag_indices = [tag_to_idx.get(tag, tag_to_idx['<UNK>']) for tag in sen]
    # Convert indices to tensor
    tag_tensor = torch.LongTensor(tag_indices)
    tags_tensors.append(tag_tensor)
    # tags_tensors=torch.stack(tags_tensors)



# Convert tag to tensors
import torch
dev_tags_tensors = []
for sen in dev_pos_tag:
    # Convert words to indices using word-to-index dictionary
    tag_indices = [tag_to_idx.get(tag, tag_to_idx['<UNK>']) for tag in sen]
    # Convert indices to tensor
    dev_tag_tensor = torch.LongTensor(tag_indices)
    dev_tags_tensors.append(dev_tag_tensor)



class LSTMTagger(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        # to the number of tag we want as output , tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
        # initialize the hidden state(see code below)
        self.hidden=self.init_hidden()
        
    def init_hidden(self):
        # the axes dimensions are (n_layers , batch_size, hidden_dim)
        return (torch.zeros(1,1,self.hidden_dim),
                torch.zeros(1,1, self.hidden_dim))
        
   
    

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out,_ = self.lstm(embeds.view(len(sentence), 1, -1),self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    
    
    
EMBEDDING_DIM=100
HIDDEN_DIM=200
epochs =10

# INSTANTIATE OUR MODEL
model=LSTMTagger(EMBEDDING_DIM,HIDDEN_DIM,len(word_to_idx), len(tag_to_idx))

# define our loss and optimizer
loss_function=nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)






from sklearn.metrics import accuracy_score
# for epoch in range(epochs):
#     print(f'On Epoch {epoch}:')
#     tr_loss=0
#     tr_acc=0
#     for i,sentence in enumerate(sentences_tensors):
#         y_pred = model(sentence)
#         loss = loss_function(y_pred,tags_tensors[i])
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         y_pred_flat = y_pred.argmax(dim=1)  # Reshape y_pred to 1D array
#         tr_acc+=accuracy_score(tags_tensors[i], y_pred_flat)
#         tr_loss+=loss.item()
#     tr_acc/=len(sentences_tensors)
#     tr_loss/=len(sentences_tensors)
#     print(f'train loss= {tr_loss}; train accuracy = {tr_acc}')
#     dev_loss, dev_acc=0,0
#     for i,sentence in enumerate(dev_sentences_tensors):
#         y_pred = model(sentence)
#         loss = loss_function(y_pred,dev_tags_tensors[i])
#         y_pred_flat = y_pred.argmax(dim=1)  # Reshape y_pred to 1D array
#         dev_acc+=accuracy_score(dev_tags_tensors[i], y_pred_flat)
#         dev_loss+=loss.item()
#     dev_acc/=len(dev_sentences_tensors)
#     dev_loss/=len(dev_sentences_tensors)
#     # dev_acc
#     print(f'dev loss= {dev_loss}; dev accuracy = {dev_acc}')
# torch.save(model,'Model.pt')


from sklearn.metrics import f1_score
import torch
test_sentences_tensors = []
for sen in test_sentence:
    # Convert words to indices using word-to-index dictionary
    word_indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in sen]
    # Convert indices to tensor
    test_sentence_tensor = torch.LongTensor(word_indices)
    test_sentences_tensors.append(test_sentence_tensor)
    
    
import torch
model=torch.load("Model.pt")

test_tags_tensors = []
for sen in test_pos_tag:
    # Convert words to indices using word-to-index dictionary
    tag_indices = [tag_to_idx.get(tag, tag_to_idx['<UNK>']) for tag in sen]
    # Convert indices to tensor
    test_tag_tensor = torch.LongTensor(tag_indices)
    test_tags_tensors.append(test_tag_tensor)
    
    

test_acc=0
test_loss=0
test_f1=0
for i,sentence in enumerate(test_sentences_tensors):
    y_pred = model(sentence)
    loss = loss_function(y_pred,test_tags_tensors[i])
    y_pred_flat = y_pred.argmax(dim=1)  # Reshape y_pred to 1D array
    test_acc+=accuracy_score(test_tags_tensors[i], y_pred_flat)
    test_f1+=f1_score(test_tags_tensors[i], y_pred_flat,average='weighted')
    test_loss+=loss.item()
test_acc/=len(test_sentences_tensors)
test_loss/=len(test_sentences_tensors)
test_f1/=len(test_sentences_tensors)
# dev_acc
print(f'test loss= {test_loss}; test accuracy = {test_acc}; test f1_score: {test_f1}')









# test_sentence="i want to fly from boston at 838 am and arive in denver at 110 in the morning".lower().split()
test_sentence=input("enter a sentence")
test_sentence=test_sentence.lower().split()
test_sentences_tensors = []

word_indices = []
for word in test_sentence:
    if word in word_to_idx.keys():
        word_indices.append(word_to_idx[word] )
    else:
        word_indices.append(word_to_idx['<UNK>'])
    

test_sentence_tensors = torch.LongTensor(word_indices)



tag_scores=model(test_sentence_tensors)
# print(tag_scores)


_, predicted_tags = torch.max(tag_scores, 1)
# print('\n')
# print('Predicted tags: \n',predicted_tags)
    
def get_key_from_value(d, val):
    for key, value in d.items():
        if value == val:
            return key
    return None  # Value not found in dictionary


for i,tag in enumerate(predicted_tags):
    print(f"{test_sentence[i]}\t{get_key_from_value(tag_to_idx,tag)}")
    
    




