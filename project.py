#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install nltk wordcloud textblob


# In[3]:


import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# 2. Loading the Datasets
trump = pd.read_csv("hashtag_donaldtrump.csv", lineterminator='\n')
print(trump.head(3))

biden = pd.read_csv("hashtag_joebiden.csv", lineterminator='\n')
print(biden.head(2))


# In[5]:


# 3. Data Assessment
print(trump.shape)
print(biden.shape)

trump.info()
biden.info()


# In[6]:


# 4. Data Preprocessing
trump['candidate'] = 'trump'
biden['candidate'] = 'biden'
data = pd.concat([trump, biden])
print('Final Data Shape :', data.shape)
print("\nFirst 2 rows:")
print(data.head(3))

# Drop any rows with missing values
data.dropna(inplace=True)

# Normalize country names
data['country'] = data['country'].replace({
    'United States of America': 'US',
    'United States': 'US'
})


# In[7]:


# 5. Exploratory Data Analysis (EDA)
# Tweets count per candidate
tweets_count = data.groupby('candidate')['tweet'].count().reset_index()
fig = px.bar(
    tweets_count, x='candidate', y='tweet', color='candidate',
    color_discrete_map={'Trump': 'pink', 'Biden': 'blue'},
    labels={'candidate': 'Candidates', 'tweet': 'Number of Tweets'},
    title='Tweets for Candidates'
)
fig.show()

# Total likes comparison
likes_comparison = data.groupby('candidate')['likes'].sum().reset_index()
fig = px.bar(
    likes_comparison, x='candidate', y='likes', color='candidate',
    color_discrete_map={'Trump': 'blue', 'Biden': 'green'},
    labels={'candidate': 'Candidate', 'likes': 'Total Likes'},
    title='Comparison of Likes'
)
fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')
fig.show()

# Top 10 tweeting countries
top10countries = (
    data.groupby('country')['tweet']
    .count()
    .sort_values(ascending=False)
    .reset_index()
    .head(10)
)
fig = px.bar(
    top10countries, x='country', y='tweet',
    template='plotly_dark',
    color_discrete_sequence=px.colors.qualitative.Dark24_r,
    title='Top10 Countrywise Tweets Counts'
)
fig.show()

# Tweet counts for each candidate in top 10 countries
tweet_df = data.groupby(['country', 'candidate'])['tweet'].count().reset_index()
tweeters = tweet_df[tweet_df['country'].isin(top10countries.country)]
fig = px.bar(
    tweeters, x='country', y='tweet', color='candidate',
    labels={'country': 'Country', 'tweet': 'Number of Tweets', 'candidate': 'Candidate'},
    title='Tweet Counts for Each Candidate in the Top 10 Countries',
    template='plotly_dark', barmode='group'
)
fig.show()


# In[8]:


# 6. Sentiment Analysis Functions
def clean(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', str(text))
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    lm = WordNetLemmatizer()
    words = [lm.lemmatize(w) for w in words if w not in set(stopwords.words('english'))]
    return ' '.join(words)

def getpolarity(text):
    return TextBlob(text).sentiment.polarity

def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getAnalysis(score):
    if score < 0:
        return 'negative'
    elif score == 0:
        return 'neutral'
    else:
        return 'positive'


# In[ ]:


# 8. Biden Tweet Sentiment Analysis
biden_tweets = data[data['candidate'] == 'biden']
biden_tweets = biden_tweets[biden_tweets.country == 'US'][['tweet']]
biden_tweets['cleantext']   = biden_tweets['tweet'].apply(clean)
biden_tweets['subjectivity']= biden_tweets['cleantext'].apply(getsubjectivity)
biden_tweets['polarity']    = biden_tweets['cleantext'].apply(getpolarity)
biden_tweets['analysis']    = biden_tweets['polarity'].apply(getAnalysis)

# Distribution plot
plt.style.use('dark_background')
colors = ['orange', 'green', 'red']
plt.figure(figsize=(7,5))
(biden_tweets.analysis.value_counts(normalize=True)*100).plot.bar(color=colors)
plt.ylabel("% of tweets")
plt.title("Distribution of Sentiments towards Biden")
plt.show()

# Word Cloud
word_cloud(biden_tweets['cleantext'][:5000])


# In[ ]:


# 9. Final Sentiment Proportions
print("Trump sentiment distribution (%):")
print(trump_tweets.analysis.value_counts(normalize=True)*100)

print("\nBiden sentiment distribution (%):")
print(biden_tweets.analysis.value_counts(normalize=True)*100)


# In[ ]:




