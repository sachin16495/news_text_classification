#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import math
from keras.layers import Activation, Conv1D, Dense, Embedding, Flatten, Input, MaxPooling1D,MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.metrics import categorical_accuracy
from keras.models import load_model


# Excel Data Frame Creation 

# In[2]:


dfexcel=pd.read_excel("news_details.xlsx")
dfexcel_cat=pd.read_excel("category_mapping.xlsx")
text_df=pd.merge(dfexcel,dfexcel_cat,on='news_id')


# Data Cleasing function clean the data fram and remove the noise from the text and also it return mean length.

# In[3]:


def text_clense_frame(text_df):
    text_frm=[]
    mid=0
    for sm in text_df:
        #print(sm)
        st=str(sm)
        if st=='nan':
            st='unk'
        smal=st.lower()
        substitutions={"'":"","(":"",")":"","+":""}
        listt=["(",")","'","-","\n"]
        for ls in listt:
            smal=smal.replace(ls,"")
        text_frm.append(smal)
    mid=mid/len(text_df)
    mid=math.floor(mid)#flor(mid)
    return text_frm


# It calculate the text train and test split size  I took 10% for testing data

# In[4]:


amout=round(len(text_df)*0.1)


# In[5]:


trainlen=len(text_df)-amout


# In[6]:


trainlen


# In[7]:


testlen=round(amout)


# In[8]:


testlen


# In[9]:


train=text_df[0:trainlen]


# In[10]:


test=text_df[trainlen:]


# In[11]:


texts = text_clense_frame(train["snippet"].values)
target = train["category_id"].values


# In[12]:


c=list(target)
target=[t-1 for t in c]


# In[13]:


texts=[str(t) for t in texts ]


# The vocabulary is depend on the total number of unique in the features 

# In[14]:


vocab_size = 9573 

tokenizer = Tokenizer(num_words=vocab_size) # Setup tokenizer
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts) # Generate sequences


# In[15]:


word_index = tokenizer.word_index


# In the below script I create inverse index mapping of each words in the feature map

# In[16]:



inv_index = {v: k for k, v in tokenizer.word_index.items()}


for w in sequences[0]:
    x = inv_index.get(w)
    print(x,end = ' ')


#  Get the average length of a text and get the standard deviation of the sequence length

# In[17]:



avg = sum(map(len, sequences)) / len(sequences)
std = np.sqrt(sum(map(lambda x: (len(x) - avg)**2, sequences)) / len(sequences))

avg,std


# Padding small sentence sequnce  

# In[18]:


max_length = 100
data = pad_sequences(sequences, maxlen=max_length)


# Generate the on hot encodding of the labeled text

# In[19]:


from keras.utils import to_categorical
labels = to_categorical(np.asarray(target),7)
print (target[0])
print (labels[0])


# 
# Load the glove embedding 6b100d encodding .I create a dictionary of word --> embedding
# 

# In[20]:


glove_dir = 'glove.6B' 

embeddings_index = {} # We create a dictionary of word -> embedding

with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0] 
        embedding = np.asarray(values[1:], dtype='float32') 
        embeddings_index[word] = embedding 


# I used 100 dimensional glove vectors. Meaning a word with token 1 needs to be in the second row (rows start with zero) and so on. If we are above the amount of words we want to use we do nothing

# In[21]:


embedding_dim = 100

word_index = tokenizer.word_index
nb_words = min(vocab_size, len(word_index)) 

embedding_matrix = np.zeros((nb_words, embedding_dim))


for word, i in word_index.items():
    if i >= vocab_size: 
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector


# In[22]:


len(embedding_matrix)


# In[23]:


print (embedding_matrix[100])


# Below you will find the architecture of the model is based on covolution neural network 1D and the activation function used in the network is relu.

# In[24]:


model = Sequential()
model.add(Embedding(vocab_size, 
                    embedding_dim, 
                    input_length=max_length, 
                    weights = [embedding_matrix]))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.summary()


# The loss I use is categorical cross entropy loss and the optimizer function is adam optimizer

# In[26]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])


# In[27]:


model.fit(data, labels, validation_split=0.2, epochs=10)


# I apply bagging on three features and by calculating 

# In[28]:


model = load_model('H5 weightCNN/snippet.h5')


# In[29]:


snipp_test=text_clense_frame(test['snippet'])
title_test=text_clense_frame(test['title'])
newdescrip_test=text_clense_frame(test['news_description'])


# In[34]:


example = text_clense_frame(snipp_test) # get the tokens

example


# In[33]:


vocab_size = 9573 

tokenizertest = Tokenizer(num_words=vocab_size) 
tokenizertest.fit_on_texts(example)
sequencestest = tokenizertest.texts_to_sequences(example) 


# In[34]:


word_index1 = tokenizertest.word_index


# In[35]:


def pridct_label(testperoc):
    pr_label=[]
    for i in testperoc:
        ind = np.unravel_index(np.argmax(i, axis=None), i.shape)
        
        pr_label.append(ind[0])
    pr_label=[(prt+1) for prt in pr_label]
    return pr_label


# I weighted average for pridicting multiple feature in the dataset.With wighted averge I calculate by the Pearson correlation coefficient with label and set that accordingly at the time of testing

# In[36]:


from scipy.stats import pearsonr


# In[37]:


Y_test=test['category_id']


# In[38]:


model.load_weights("H5 weightCNN/snippet.h5")


# In[39]:


snippetlabel = model.predict(x_train_pad)


# In[40]:


titlecoff=np.nan_to_num(pearsonr(snippetlabel, Y_train)[0])


# In[41]:


titlecoff


# In[43]:


model.load_weights("H5 weightCNN/news_description.h5")


# In[44]:


descriplabel = model.predict(x_train_pad)


# In[45]:


descripcoff=np.nan_to_num(pearsonr(descriplabel, Y_train)[0])


# In[46]:


descripcoff


# In[47]:


model.load_weights("H5 weightCNN/title.h5")


# In[48]:


descriplabel = model.predict(x_train_pad)


# In[49]:


descripcoff=np.nan_to_num(pearsonr(descriplabel, Y_train)[0])


# In[50]:


Y_test=[tes for tes in Y_test]


# In[51]:


max_length = 100
datatest = pad_sequences(sequencestest, maxlen=max_length)


# In[52]:


snipptest=X_test['snippet']
titletest=X_test['title']
descrip_test=X_test['news_description']


# In[53]:


model.load_weights("H5 weightCNN/modeltsnippt.h5")


# In[54]:


num_words1=100
Tokenize the text
tokenize1=Tokenizer(num_words=num_words1)
tokenize1.fit_on_texts(X_train)
idx=tokenize.word_index
x_test_tokensnip=tokenize.texts_to_sequences(snipptest)


# In[55]:


pad='pre'
x_test_padsnip=pad_sequences(x_test_tokensnip,maxlen=10,padding=pad,truncating=pad)


# In[56]:


predtest1 = model.predict(x_test_padsnip)


# In[57]:


sniptestplabels=pridct_label(predtest1)


# In[58]:


model.load_weights("H5 weightCNN/modeltitle.h5")


# In[59]:


num_words2=100
Tokenize the text
tokenize2=Tokenizer(num_words=num_words2)
tokenize2.fit_on_texts(titletest)
idx=tokenize.word_index
x_test_tokentitle=tokenize.texts_to_sequences(titletest)
pad='pre'
x_test_title_token=pad_sequences(x_test_tokentitle,maxlen=10,padding=pad,truncating=pad)


# In[60]:


x_test_modeltitle=pad_sequences(x_test_title_token,maxlen=10,padding=pad,truncating=pad)


# In[61]:


predtest2=model.predict(x_test_modeltitle)


# In[62]:


titletestplabels=pridct_label(predtest2)


# In[63]:


model.load_weights("H5 weightCNN/descrips_text.h5")


# In[64]:


num_words3=100
Tokenize the text
tokenize3=Tokenizer(num_words=num_words3)
tokenize3.fit_on_texts(descrip_test)
idx=tokenize.word_index
x_test_tokentitle=tokenize.texts_to_sequences(descrip_test)
pad='pre'
x_test_descrip_token=pad_sequences(x_test_tokentitle,maxlen=10,padding=pad,truncating=pad)


# In[66]:


predtest3 = model.predict(x_test_descrip_token)


# In[67]:


desctesplabels=pridct_label(predtest3)


# In[68]:


def scarmul(lis1,elem,lis2,elem1,lis3,elem3):
    listfi=[]
    for a,b,c in zip(lis1,lis2,lis3):
        cal=round((a*elem)+(b*elem1)+(c*elem3))
        listfi.append(cal)
    return listfi


# In[73]:


prff=scarmul(sniptestplabels,snipcoff,titletestplabels,titlecoff,desctesplabels,desccoff)


# In[74]:


from sklearn.metrics import f1_score


# I used F1 score to evaluate this model which give the result below

# In[77]:


f1_score(Y_test, prtf,average='macro')  


# In[78]:


from sklearn.metrics import roc_auc_score


# In[79]:


from sklearn.preprocessing import LabelBinarizer


# In[80]:


lb = LabelBinarizer()
lb.fit(Y_test)
y_test = lb.transform(Y_test)
y_pred = lb.transform(prff)


# In[81]:


roc_auc_score(y_test, y_pred, average='macro')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




