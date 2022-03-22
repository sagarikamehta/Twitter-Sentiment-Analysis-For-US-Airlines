#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:





# In[3]:





# In[4]:


import os
import re
import nltk
from nltk.corpus import stopwords
del data['Tweet_Coordinates']
del data['Airline_Sentiment_Gold']
del data['Negative_Reason_Gold']
data.head()


# In[5]:


import matplotlib.pyplot as plt
print("Total number of tweets for each airline \n ",data.groupby('Airline')['Airline_Sentiment'].count().sort_values(ascending=False))
airlines= ['US Airways','United','American','Southwest','Delta','Virgin America']
plt.figure(1,figsize=(12, 12))
for i in airlines:
    indices= airlines.index(i)
    plt.subplot(2,3,indices+1)
    new_data=data[data['Airline']==i]
    count=new_data['Airline_Sentiment'].value_counts()
    Index = [1,2,3]
    plt.bar(Index,count, color=['red', 'green', 'blue'])
    plt.xticks(Index,['negative','neutral','positive'])
    plt.ylabel('Mood Count')
    plt.xlabel('Mood')
    plt.title('Count of Moods of '+i)


# In[6]:


data['Negative_Reason'].nunique()
NR_Count=dict(data['Negative_Reason'].value_counts(sort=False))
def NR_Count(Airline):
    if Airline=='All':
        a=data
    else:
        a=data[data['Airline']==Airline]
    count=dict(a['Negative_Reason'].value_counts())
    Unique_reason=list(data['Negative_Reason'].unique())
    Unique_reason=[x for x in Unique_reason if str(x) != 'nan']
    Reason_frame=pd.DataFrame({'Reasons':Unique_reason})
    Reason_frame['count']=Reason_frame['Reasons'].apply(lambda x: count[x])
    return Reason_frame
def plot_reason(Airline):
    a=NR_Count(Airline)
    count=a['count']
    Index = range(1,(len(a)+1))
    plt.bar(Index,count, color=['red','yellow','blue','green','black','brown','gray','cyan','purple','orange'])
    plt.xticks(Index,a['Reasons'],rotation=90)
    plt.ylabel('Count')
    plt.xlabel('Reason')
    plt.title('Count of Reasons for '+Airline)
plot_reason('All')
plt.figure(2,figsize=(13, 13))
for i in airlines:
    indices= airlines.index(i)
    plt.subplot(2,3,indices+1)
    plt.subplots_adjust(hspace=0.9)
    plot_reason(i)


# In[7]:


date = data.reset_index()
date.Tweet_Created = pd.to_datetime(date.Tweet_Created)
date.Tweet_Created = date.Tweet_Created.dt.date
date.Tweet_Created.head()
data = date
day_data = data.groupby(['Tweet_Created','Airline','Airline_Sentiment']).size()
day_data


# In[8]:


day_data = day_data.loc(axis=0)[:,:,'negative']
ax2 = day_data.groupby(['Tweet_Created','Airline']).sum().unstack().plot(kind = 'bar', color=['red', 'green', 'blue','yellow','purple','orange'], figsize = (15,6), rot = 70)
labels = ['American','Delta','Southwest','US Airways','United','Virgin America']
ax2.legend(labels = labels)
ax2.set_xlabel('Date')
ax2.set_ylabel('Negative Tweets')
plt.show()


# In[10]:


import nltk
from nltk.corpus import stopwords
def tweet_to_words(tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words ))


# In[11]:


data['Clean_Tweet']=data['Text'].apply(lambda x: tweet_to_words(x))


# In[12]:


from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size=0.2,random_state=42)


# In[13]:


train_clean_tweet=[]
for tweet in train['Clean_Tweet']:
    train_clean_tweet.append(tweet)
test_clean_tweet=[]
for tweet in test['Clean_Tweet']:
    test_clean_tweet.append(tweet)


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer(analyzer = "word")
train_features= v.fit_transform(train_clean_tweet)
test_features=v.transform(test_clean_tweet)


# In[15]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
Classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200)]


# In[17]:


import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
dense_features=train_features.toarray()
dense_test= test_features.toarray()
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['Airline_Sentiment'])
        pred = fit.predict(test_features)
    except Exception:
        fit = classifier.fit(dense_features,train['Airline_Sentiment'])
        pred = fit.predict(dense_test)
    accuracy = accuracy_score(pred,test['Airline_Sentiment'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))
    print(classification_report(pred,test['Airline_Sentiment']))
    cm=confusion_matrix(pred , test['Airline_Sentiment'])
    plt.figure()
    plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Reds)
    plt.xticks(range(2), ['Negative', 'Neutral', 'Positive'], fontsize=16,color='black')
    plt.yticks(range(2), ['Negative', 'Neutral', 'Positive'], fontsize=16)
    plt.show()


# In[ ]:




