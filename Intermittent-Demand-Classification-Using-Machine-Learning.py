#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib nbagg
from sklearn.naive_bayes import *
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:


data = {'Sales': [0,0,0,0,0,0,0,0,1,1,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,1,0,0
                  ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'dayOfWeek': [5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,
                      6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2],
         'promo': [10,20,15,10,15,10,20,10,20,15,10,15,10,20,10,20,15,10,15,10,20,10,20,15,10,15,10,20,10,20,
                   15,10,15,10,20,10,20,15,10,15,10,20,10,20,15,10,15,10,20,10,20,15,10,15,10,20,10,20,15,10,15],
        'marketing' :[1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,
                      0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0]
        }  


# In[4]:


df = pd.DataFrame(data, columns = ['Sales', 'dayOfWeek', 'promo', 'marketing'])


# In[5]:


df.head()


# In[25]:


df.describe().transpose()


# In[6]:


#Co-relation Matrix

corr_mat=df.corr()

fig = plt.figure()
sns.set(rc={'figure.figsize':(15,10)})
k =10
cols=corr_mat.nlargest(k, 'Sales')['Sales'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm=sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10}, 
               yticklabels=cols.values, xticklabels=cols.values)
plt.title=('Correlation Matrix')
plt.show()


# In[7]:


corr_mat['Sales'].sort_values(ascending=False)


# In[8]:


print ('Count of Sales')
print(df.loc[df["Sales"]==1].groupby("dayOfWeek").size())


# In[9]:


s = df.Sales
plt.figure()
plt.plot(s)
#plt.ylim(0, 40)`


# In[32]:


#split test and train data

X = df.drop('Sales', axis=1)
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[33]:


X_train.head()
X_test.head()
y_train.head()
y_test.head()


# # Naive Bayes

# In[34]:


modelGNB = GaussianNB()


# In[35]:


modelGNB.fit(X_train, y_train)


# In[36]:


predGNB = modelGNB.predict(X_test)
predGNB


# In[37]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[38]:


# classification report for precision, recall f1-score and accuracy
print(predGNB.size)
print('Positive cases:', y_test[y_test==1].shape[0])
print('Negative cases:', y_test[y_test==0].shape[0])

print('Accuracy Score')
print(accuracy_score(y_test, predGNB))
print('Precision/Recall Metrics')
print(classification_report(y_test, predGNB))

print('\n')
print(confusion_matrix(y_test, predGNB))


# In[39]:


print('Confusion Matrix')
cm =confusion_matrix(y_test, predGNB)
lbl1=['Predicted 0', 'Predicted 1']
lbl2=['True 0', 'True 1']

sns.heatmap(cm, annot=True, cmap="Greens", fmt='d', xticklabels = lbl1, yticklabels=lbl2)


# In[40]:


from sklearn.metrics import precision_score, recall_score
precision_score(y_test,predGNB)
recall_score(y_test,predGNB)


# # SVM Classification

# In[41]:


from sklearn.svm import SVC                   # "Support Vector Classifier" 
modelSVC = SVC(kernel='linear') 


# In[42]:


modelSVC.fit(X_train, y_train)


# In[43]:


predSVC= modelSVC.predict(X_test)


# In[44]:


# classification report for precision, recall f1-score and accuracy
print(predSVC.size)
print('Positive cases:', y_test[y_test==1].shape[0])
print('Negative cases:', y_test[y_test==0].shape[0])

print('Accuracy Score')
print(accuracy_score(y_test, predSVC))
print('Precision/Recall Metrics')
print(classification_report(y_test, predSVC))

print('\n')
print(confusion_matrix(y_test, predSVC))


# In[45]:


print('Confusion Matrix')
cm =confusion_matrix(y_test, predSVC)
lbl1=['Predicted 0', 'Predicted 1']
lbl2=['True 0', 'True 1']

sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels = lbl1, yticklabels=lbl2)


# # Neural Network Classifier

# In[46]:


from sklearn.neural_network import MLPClassifier
modelMLP = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)


# In[47]:


modelMLP.fit(X_train, y_train)


# In[48]:


predMLP= modelMLP.predict(X_test)


# In[49]:


# classification report for precision, recall f1-score and accuracy
print(predMLP.size)
print('Positive cases:', y_test[y_test==1].shape[0])
print('Negative cases:', y_test[y_test==0].shape[0])

print('Accuracy Score')
print(accuracy_score(y_test, predMLP))
print('Precision/Recall Metrics')
print(classification_report(y_test, predMLP))

print('\n')
print(confusion_matrix(y_test, predMLP))


# In[89]:


print('Confusion Matrix')
cm =confusion_matrix(y_test, predMLP)
lbl1=['Predicted 0', 'Predicted 1']
lbl2=['True 0', 'True 1']

sns.heatmap(cm, annot=True, cmap="Reds", fmt='d', xticklabels = lbl1, yticklabels=lbl2)


# # Creating Additional Features- Feature Engineering

# # Rolling Window

# In[51]:


data = df['Sales'].values
data


# In[52]:


#Create Rolling Window  
def rolling_window(data, win_length):
    rw_df = []
    win_len = win_length
    win_count = len(data)# - win_len
    for count in range(win_count):
        print(count, win_count)
        index = count + win_len
        print(data[count:index])
        rw_df.append(data[count:index])
    return rw_df


# In[53]:


rw = rolling_window(data, 7)


# In[54]:


df2 = pd.DataFrame(rw, columns = ('Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'))
pd.set_option('display.max_rows', None)
df2


# In[55]:


df2['Demand_Days'] = df2['Day 1'] + df2['Day 2'] + df2['Day 3'] + df2['Day 4'] + df2['Day 5'] + df2['Day 6'] + df2['Day 7']


# In[56]:


df2['No_Demand_Days'] = 7-df2['Demand_Days']


# In[57]:


df2 = pd.concat([df, df2], axis=1)


# In[58]:


#Create Lag VAriables
df2['Sales_lag1'] = df2['Sales'].shift(1)
df2['Sales_lag2'] = df2['Sales'].shift(2)
df2['Sales_lag3'] = df2['Sales'].shift(3)
df2['Sales_lag4'] = df2['Sales'].shift(4)
df2['Sales_lag5'] = df2['Sales'].shift(5)
df2['Sales_lag6'] = df2['Sales'].shift(6)
df2['Sales_lag6'] = df2['Sales'].shift(6)


# In[59]:


df2['Rolling_Mean3'] = df2['Sales'].rolling(window=3).mean()
df2['Rolling_Mean6'] = df2['Sales'].rolling(window=6).mean()


# In[60]:


countSales = -1;
nCol = []
for element in range(len(df)):
    if(df['Sales'][element] == 0):
        countSales =countSales+1;
        nCol.append(countSales)
    else:
        if(countSales != -1):
            countSales =countSales+1;
            nCol.append(countSales)
        else:
            nCol.append(0)
        countSales=-1;
        
df2['ZeroCumulative'] = nCol


# In[61]:


df2 = df2.drop('Day 1', axis=1)


# In[62]:


df2=df2.fillna(0)


# In[73]:


#split test and train data

X = df2.drop('Sales', axis=1)
y = df2['Sales']
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[74]:


X2_train.head()
X2_test.head()
y2_train.head()
y2_test.head()


# # Naive Bayes-With Features

# In[75]:


modelGNB2 = GaussianNB()
modelGNB2.fit(X2_train, y2_train)


# In[76]:


predGNB2 = model2.predict(X2_test)
predGNB2


# In[77]:


# classification report for precision, recall f1-score and accuracy
print(predGNB2.size)
print('Positive cases:', y2_test[y2_test==1].shape[0])
print('Negative cases:', y2_test[y2_test==0].shape[0])

print('Accuracy Score')
print(accuracy_score(y2_test, predGNB2))
print('Precision/Recall Metrics')
print(classification_report(y2_test, predGNB2))

print('\n')
print(confusion_matrix(y2_test, predGNB2))


# In[78]:


print('Confusion Matrix')
cm =confusion_matrix(y2_test, predGNB2)
lbl1=['Predicted 0', 'Predicted 1']
lbl2=['True 0', 'True 1']

sns.heatmap(cm, annot=True, cmap="Greens", fmt='d', xticklabels = lbl1, yticklabels=lbl2)


# # SVM Classification -With Features

# In[79]:


from sklearn.svm import SVC                   # "Support Vector Classifier" 
modelSVC2 = SVC(kernel='linear') 


# In[80]:


modelSVC2.fit(X2_train, y2_train)


# In[81]:


predSVC2= modelSVC2.predict(X2_test)


# In[82]:


# classification report for precision, recall f1-score and accuracy
print(predSVC2.size)
print('Positive cases:', y2_test[y2_test==1].shape[0])
print('Negative cases:', y2_test[y2_test==0].shape[0])

print('Accuracy Score')
print(accuracy_score(y2_test, predSVC2))
print('Precision/Recall Metrics')
print(classification_report(y2_test, predSVC2))

print('\n')
print(confusion_matrix(y2_test, predSVC2))


# In[83]:


print('Confusion Matrix')
cm =confusion_matrix(y2_test, predSVC2)
lbl1=['Predicted 0', 'Predicted 1']
lbl2=['True 0', 'True 1']

sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels = lbl1, yticklabels=lbl2)


# # Neural Network Classifier-With Features

# In[84]:


from sklearn.neural_network import MLPClassifier
modelMLP2 = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)


# In[90]:


modelMLP2.fit(X2_train, y2_train)


# In[91]:


predMLP2= modelMLP2.predict(X2_test)


# In[92]:


# classification report for precision, recall f1-score and accuracy
print(predMLP2.size)
print('Positive cases:', y2_test[y2_test==1].shape[0])
print('Negative cases:', y2_test[y2_test==0].shape[0])

print('Accuracy Score')
print(accuracy_score(y2_test, predMLP2))
print('Precision/Recall Metrics')
print(classification_report(y2_test, predMLP2))

print('\n')
print(confusion_matrix(y2_test, predMLP2))


# In[93]:


print('Confusion Matrix')
cm =confusion_matrix(y2_test, predMLP2)
lbl1=['Predicted 0', 'Predicted 1']
lbl2=['True 0', 'True 1']

sns.heatmap(cm, annot=True, cmap="Reds", fmt='d', xticklabels = lbl1, yticklabels=lbl2)


# In[ ]:





# # Using KFold Cross Validation 

# In[37]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict 


# In[38]:


X = df.drop('Sales', axis=1)
y = df['Sales']


# In[39]:


X2 = df2.drop('Sales', axis=1)
y2 = df2['Sales']


# In[40]:


# prepare cross validation
kfold = KFold(n_splits=5, random_state=42, shuffle=False)


# Without Feature Engineering

# In[41]:


my_scores = [] 

for train, test in kfold.split(df):
    modelk = GaussianNB()
    modelk.fit(X, y)
    my_scores.append(modelk.score(X, y)) 
     
print("The mean value is" ) 
print(np.mean(my_scores)) 


# In[42]:


scores = cross_val_score(modelk,X, y, cv=5)
scores


# In[43]:


KPred = cross_val_predict(modelk,X, y, cv=5)
KPred


# In[44]:


# classification report for precision, recall f1-score and accuracy
print(KPred.size)
print('Positive cases:', y_test[y_test==1].shape[0])
print('Negative cases:', y_test[y_test==0].shape[0])

print('Accuracy Score')
print(accuracy_score(y, KPred))
print('Precision/Recall Metrics')
print(classification_report(y, KPred))

print('\n')
print(confusion_matrix(y, KPred))


# In[45]:


print('Confusion Matrix')
cm =confusion_matrix(y, KPred)
lbl1=['Predicted 0', 'Predicted 1']
lbl2=['True 0', 'True 1']

sns.heatmap(cm, annot=True, cmap="Greens", fmt='d', xticklabels = lbl1, yticklabels=lbl2)


# With Feature Engineering

# In[46]:


my_scores = [] 

for train, test in kfold.split(df2):
    modelk = GaussianNB()
    modelk.fit(X2, y2)
    my_scores.append(modelk.score(X2, y2)) 
     
print("The mean value is" ) 
print(np.mean(my_scores)) 


# In[47]:


scores = cross_val_score(modelk,X2, y2, cv=5)
scores


# In[48]:


KPred = cross_val_predict(modelk,X2, y2, cv=5)
KPred


# In[49]:


# classification report for precision, recall f1-score and accuracy
print(KPred.size)
print('Positive cases:', y2_test[y2_test==1].shape[0])
print('Negative cases:', y2_test[y2_test==0].shape[0])

print('Accuracy Score')
print(accuracy_score(y2, KPred))
print('Precision/Recall Metrics')
print(classification_report(y2, KPred))

print('\n')
print(confusion_matrix(y2, KPred))


# In[50]:


print('Confusion Matrix')
cm =confusion_matrix(y2, KPred)
lbl1=['Predicted 0', 'Predicted 1']
lbl2=['True 0', 'True 1']

sns.heatmap(cm, annot=True, cmap="Greens", fmt='d', xticklabels = lbl1, yticklabels=lbl2)


# # Using Croston's method
# 

# In[51]:


from croston import croston


# In[52]:


y = df['Sales']
y


# In[53]:


fit_pred = croston.fit_croston(y_train, 21,'original')


# In[54]:


yhat_croston = pd.DataFrame(np.concatenate([fit_pred['croston_fittedvalues'], fit_pred['croston_forecast']]))
yhat_croston

