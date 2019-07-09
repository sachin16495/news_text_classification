#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import sys
from keras.utils import np_utils
import numpy as np
from sklearn import model_selection
from scipy.spatial.distance import cdist
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.layers import LSTM,Dense,GRU,Embedding,SpatialDropout1D
import pandas as pd
import math


# The function below responsible for cleasning the text by replacing symbols and handling nana

# In[2]:


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
        #print(smal)
        splitext=smal.split(' ')
        #splitext=splitext.split('\n')
        mid=mid+len(splitext)
        text_frm.append(splitext)
    mid=mid/len(text_df)
    mid=math.floor(mid)#flor(mid)
    return text_frm,mid


# In[3]:


def pad_clip(dafrm,maxlen):
    padded_frame=[]
    pword_frame=[]
    co=0
    
    for df in dafrm:
        lendf=len(df)
        pword_frame=[]
        if lendf<maxlen:
            pad_len=maxlen-lendf
            pword_frame=df
            for pa in range(0,pad_len):
                pword_frame.extend('p')
        else:
            pword_frame=df[:maxlen]
            #print(pword_frame)
            co=co+1
        padded_frame.append(pword_frame)
             
    return padded_frame


# In[4]:



import matplotlib.pyplot as plt


# In[5]:


dfexcel=pd.read_excel("news_details.xlsx")
dfexcel_cat=pd.read_excel("category_mapping.xlsx")
text_df=pd.merge(dfexcel,dfexcel_cat,on='news_id')


# In[6]:


amout=round(len(text_df)*0.1)


# In[7]:


trainlen=len(text_df)-amout


# In[8]:


testlen=round(amout)


# Train Test split of dataframe

# In[9]:


X_train,X_test =text_df[:trainlen],text_df[trainlen:]# model_selection.train_test_split(X,Y,test_size=0.30)


# In[10]:


len(X_train)


# In[11]:


X=text_df['snippet']
Y=text_df['category_id']


# In[12]:


X_test=text_df['snippet']


# In[13]:


len(X)


# In[14]:


Y_train=X_train['category_id']


# In[15]:


X,mid=text_clense_frame(X)


# Tokenization of words

# In[16]:


num_words=mid
#Tokenize the text
tokenize=Tokenizer(num_words=num_words)
tokenize.fit_on_texts(X)
idx=tokenize.word_index
x_train_token=tokenize.texts_to_sequences(X)
#x_test_token=tokenize.texts_to_sequences(X_test)


# In[18]:


num_tokens=[len(token) for token in x_train_token]
num_tokens=np.array(num_tokens)
max_tokens=np.mean(num_tokens)+2*np.std(num_tokens)
max_tokens=int(max_tokens)
print("Max Tokens")
print(max_tokens)


# In[19]:


c=list(Y_train)


# In[20]:


lis=[t-1 for t in c]


# Categorical conversion of labels

# In[21]:


catres_label=np_utils.to_categorical(lis, 7)


# Pad the sequnce with low length

# In[23]:


pad='pre'
x_train_pad=pad_sequences(x_train_token,maxlen=max_tokens,padding=pad,truncating=pad)


# In[24]:


len(x_train_pad)


# Model architecture with lstm and and embedding layer

# In[25]:


reuse_word = 200
#Model Architecture
model = Sequential()
model.add(Embedding(reuse_word, 64, input_length=max_tokens))
model.add(LSTM(64, dropout=0.2))
model.add(Dense(7, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()


# Uncomment below to start training model

# In[64]:


#model.fit(x_train_pad, catres_label, epochs=100, batch_size=100,validation_split=0.25)


# In[26]:


def pridct_label(testperoc):
    pr_label=[]
    for i in testperoc:
        ind = np.unravel_index(np.argmax(i, axis=None), i.shape)
        
        pr_label.append(ind[0])
    pr_label=[(prt+1) for prt in pr_label]
    return pr_label


# In[27]:


model.load_weights("H5 weightLSTM/modeltsnippt.h5")


# In[28]:


pred = model.predict(x_train_pad)


# In[29]:


sniptrainlabel=pridct_label(pred)


# In[30]:


model.load_weights("H5 weightLSTM/descrips_text.h5")


# In[31]:


pred1 = model.predict(x_train_pad)


# In[32]:


descptrainlabel=pridct_label(pred1)


# In[33]:


model.load_weights("H5 weightLSTM/modeltitle.h5")
pred2 = model.predict(x_train_pad)
titlelabel=pridct_label(pred2)


# Use the bagging with Pearson correlation coefficient and apply weighted average on the test prediction 

# In[34]:


from scipy.stats import pearsonr


# In[35]:


len(Y_train)


# In[37]:


titlecoff=np.nan_to_num(pearsonr(titlelabel, Y_train)[0])


# In[41]:


titlecoff


# In[40]:


snipcoff=np.nan_to_num(pearsonr(sniptrainlabel, Y_train)[0])


# In[42]:


snipcoff


# In[43]:


desccoff=np.nan_to_num(pearsonr(descptrainlabel,Y_train)[0])


# In[44]:


desccoff


# In[47]:


snipptest=X_test['snippet']
titletest=X_test['title']
descrip_test=X_test['news_description']


# In[48]:


model.load_weights("H5 weightLSTM/modeltsnippt.h5")


# In[49]:


num_words1=100
#Tokenize the text
tokenize1=Tokenizer(num_words=num_words1)
tokenize1.fit_on_texts(X_train)
idx=tokenize.word_index
x_test_tokensnip=tokenize.texts_to_sequences(snipptest)


# In[50]:


pad='pre'
x_test_padsnip=pad_sequences(x_test_tokensnip,maxlen=10,padding=pad,truncating=pad)


# In[51]:


predtest1 = model.predict(x_test_padsnip)


# In[52]:


sniptestplabels=pridct_label(predtest1)


# In[53]:


model.load_weights("H5 weightLSTM/modeltitle.h5")


# In[54]:


num_words2=100
tokenize2=Tokenizer(num_words=num_words2)
tokenize2.fit_on_texts(titletest)
idx=tokenize.word_index
x_test_tokentitle=tokenize.texts_to_sequences(titletest)
pad='pre'
x_test_title_token=pad_sequences(x_test_tokentitle,maxlen=10,padding=pad,truncating=pad)


# In[55]:


x_test_modeltitle=pad_sequences(x_test_title_token,maxlen=10,padding=pad,truncating=pad)


# In[56]:


predtest2=model.predict(x_test_modeltitle)


# In[57]:


titletestplabels=pridct_label(predtest2)


# In[58]:


model.load_weights("H5 weightLSTM/descrips_text.h5")


# In[59]:


num_words3=100
Tokenize the text
tokenize3=Tokenizer(num_words=num_words3)
tokenize3.fit_on_texts(descrip_test)
idx=tokenize.word_index
x_test_tokentitle=tokenize.texts_to_sequences(descrip_test)
pad='pre'
x_test_descrip_token=pad_sequences(x_test_tokentitle,maxlen=10,padding=pad,truncating=pad)


# In[60]:


predtest3 = model.predict(x_test_pad)


# In[61]:


desctesplabels=pridct_label(predtest3)


# In[62]:


import math


# In[63]:


def scarmul(lis1,elem,lis2,elem1,lis3,elem3):
    listfi=[]
    for a,b,c in zip(lis1,lis2,lis3):
        cal=round((a*elem)+(b*elem1)+(c*elem3))
        listfi.append(cal)
    return listfi


# In[64]:


predicted_rest=scarmul(sniptestplabels,snipcoff,titletestplabels,titlecoff,desctesplabels,desccoff)


# In[65]:


from sklearn.metrics import f1_score


# In[66]:


f1_score(Y_test, predicted_rest,average='macro')  


# In[67]:


from sklearn.metrics import confusion_matrix


# In[68]:


from sklearn.metrics import roc_auc_score


# In[69]:


from sklearn.preprocessing import LabelBinarizer


# In[71]:


lb = LabelBinarizer()
lb.fit(Y_test)
y_test = lb.transform(Y_test)
y_pred = lb.transform(predicted_rest)


# In[72]:


roc_auc_score(y_test, y_pred, average='macro')


# In[ ]:




