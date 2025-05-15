#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,f1_score, confusion_matrix, precision_score, recall_score


# In[26]:


raw_mail_data = pd.read_csv('mail_data.csv')


# In[27]:


print(raw_mail_data)


# In[28]:


mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[29]:


mail_data.head()


# In[30]:


# checking the number of rows and columns in the dataframe
mail_data.shape


# In[31]:


# label spam mail as 0;  ham mail as 1;

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# In[32]:


# separating the data as texts and label

X = mail_data['Message']

Y = mail_data['Category']


# In[33]:


print(X)


# In[34]:


print(Y)


# In[35]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[36]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[37]:


# transform the text data to feature vectors that can be used as input to the Logistic regression

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
vectorizer = TfidfVectorizer(lowercase=True)




# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[38]:


print(X_train)


# In[39]:


print(X_train_features)


# In[40]:


model = LogisticRegression()


# In[41]:


# training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)


# In[42]:


prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
f1_score_training_data = f1_score(Y_train, prediction_on_training_data)
confusion_matrix_training_data = confusion_matrix(Y_train, prediction_on_training_data)
precision_score_training_data = precision_score(Y_train, prediction_on_training_data)
recall_score_training_data = recall_score(Y_train, prediction_on_training_data)


# In[43]:


print('Accuracy on training data : ', accuracy_on_training_data)
print('F1 score on training data: ', f1_score_training_data)
print('Confusion matrix on training data:\n', confusion_matrix_training_data)
print('Precision score on training data: ', precision_score_training_data)
print('Recall score on training data: ', recall_score_training_data)


# In[44]:


prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
f1_score_test_data = f1_score(Y_test, prediction_on_test_data)
confusion_matrix_test_data = confusion_matrix(Y_test, prediction_on_test_data)
precision_score_test_data = precision_score(Y_test, prediction_on_test_data)
recall_score_test_data = recall_score(Y_test, prediction_on_test_data)


# In[45]:


print('Accuracy on test data : ', accuracy_on_test_data)
print('F1 score on test data: ', f1_score_test_data)
print('Confusion matrix on test data:\n', confusion_matrix_test_data)
print('Precision score on test data: ', precision_score_test_data)
print('Recall score on test data: ', recall_score_test_data)


# In[46]:


input_mail = ["URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')


# In[ ]:




