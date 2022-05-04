import streamlit as st
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px
import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import os,sys
from PIL import Image



st.title('Friends dataset')
st.image('friends.jpeg')

if st.button('Say hello'):
     st.write('Welcome to Friends sitcom EDA analysis')
else:
     pass

#####Loading the data#######

df = pd.read_csv('friends_episodes_v3.csv')

pd.set_option('display.max_colwidth', None)


df2 = pd.DataFrame({'Episodes': df.Episode_Title.count(), 'Avg Episode duration': df.Duration.mean(), 'Seasons': df.Season.max(), 'Highest Star rated': df.Stars.max(),
    'Productions Years': ['1994 - 2004'],'Directors': df.Director.count()}, index = None)
df2 = df2.reset_index(drop=True)

st.write(df2)


st.header('Dataset overview')


add_selectbox = st.sidebar.selectbox(
    "Would you like to read episode summary?",
    sorted(df['Summary'].unique())
)
st.write('Your choosen episode summary:')
st.write(add_selectbox)




###PLOTS####


#creating mean of stars based on year of production
mean_col = df.groupby(by='Year_of_prod')["Stars"].mean() # don't reset the index!
df = df.set_index(['Year_of_prod']) # make the same index here
df['mean_col'] = mean_col
df = df.reset_index() # to take the hierarchical index off again
df.head()
#creating mean of votes based on the season
mean_vote = df.groupby(by ='Season')["Votes"].mean() # don't reset the index!
df = df.set_index(['Season']) # make the same index here
df['mean_votes'] = mean_vote
df = df.reset_index() # to take the hierarchical index off again
df


st.subheader('Average Ratings and Votes')
fig = go.Figure(data=go.Scatter(x=df.Season, y=df.mean_votes, mode='lines+markers', line=dict(color="lightblue")))
fig.update_xaxes(title='Season', showline=True, linewidth=2, linecolor='black', gridcolor='lightgrey')
fig.update_yaxes(title='Average Votes', showline=True, linewidth=2, linecolor='black',gridcolor='lightgrey')
fig.update_layout(title_text='Average Votes by Season', title_x=0.5, xaxis = dict(
        tickmode = 'array',
        tickvals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
fig.update_traces(marker=dict(
            color='lightblue',
            size=2,
            line=dict(
                color='darkslategray',
                width=8
            )))

fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })

fig.update_layout(
    width=800,
    height=400,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=25
        )
    )
st.plotly_chart(fig)

fig = go.Figure(data=go.Scatter(x=df.Year_of_prod, y=df.mean_col, mode='lines+markers', line=dict(color="lightblue")))
fig.update_xaxes(title='Year', showline=True, linewidth=2, linecolor='black', gridcolor='lightgrey')
fig.update_yaxes(title='Average Star Rating', showline=True, linewidth=2, linecolor='black',gridcolor='lightgrey')
fig.update_layout(title_text='Average Star Rating by Year', title_x=0.5)
fig.update_traces(marker=dict(
            color='lightblue',
            size=2,
            line=dict(
                color='darkslategray',
                width=8
            )))

fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

fig.update_layout(
    width=800,
    height=400,
    margin=dict(
        l=0,
        r=0,
        b=70,
        t=25
        )
    )
st.plotly_chart(fig)


st.subheader('Correlation')
fig = px.scatter(df, x="Stars", y="Votes", color="Season")
fig.update_xaxes(title='Average Stars', showline=True, linewidth=2, linecolor='black', gridcolor='lightgrey')
fig.update_yaxes(title='Average Votes', showline=True, linewidth=2, linecolor='black', gridcolor='lightgrey')
fig.update_layout(title_text='Average Votes and Stars correlation by Season', title_x=0.5)
fig.update_traces(marker=dict(
            color= df.Season,
            size=12,
            line=dict(
                width=2
             )))

fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
     'paper_bgcolor': 'rgba(0, 0, 0, 0)'
     })
#fig.update_layout(showlegend=True)
fig.update_layout(
    width=800,
    height=400,
    margin=dict(
        l=0,
        r=0,
        b=70,
        t=25
        )
    )
st.plotly_chart(fig)

##############################################


#setting up the tf-idf 
X= df.Director
tf_trigrams = TfidfVectorizer(ngram_range=(2, 2), stop_words='english')
tf_trigrams.fit(X)
trigrams_tf = tf_trigrams.transform(X)
trigrams_df = pd.DataFrame(trigrams_tf.todense(), columns=tf_trigrams.get_feature_names())
trigrams_df.sum().sort_values(ascending=False).head(30)

top_texts3 = trigrams_df.sum().sort_values(ascending=False)
top_texts3.head(15).plot(kind='barh')# Create and generate a word cloud image:
Cloud = WordCloud( width = 600, height = 400, 
                background_color = 'white',
                stopwords = 'English', 
                min_font_size = 3,
                min_word_length=0).generate_from_frequencies(top_texts3) 

X1= df.Episode_Title
tf_trigrams1 = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tf_trigrams1.fit(X1)
trigrams_tf1 = tf_trigrams1.transform(X1)
trigrams_df1 = pd.DataFrame(trigrams_tf1.todense(), columns=tf_trigrams1.get_feature_names())
trigrams_df1.sum().sort_values(ascending=False).head(30)

top_texts2 = trigrams_df1.sum().sort_values(ascending=False)
top_texts2.head(15).plot(kind='barh')# Create and generate a word cloud image:
Cloud2 = WordCloud( width = 600, height = 400, 
                background_color ='white',
                stopwords = 'English', 
                min_font_size = 3,
                min_word_length=0).generate_from_frequencies(top_texts2)

###wordclouds#########

cloud1, cloud2 = st.columns((2))

with cloud1:
    st.subheader('Top directors')
    fig, ax = plt.subplots()
    #plt.figure(figsize=[15,10])
    plt.imshow(Cloud, interpolation='bilinear')
    plt.axis("off")
    #plt.title('Top directors')
    st.pyplot(fig)

with cloud2:
    st.subheader('Top words in episode titles')
    fig, ax = plt.subplots()
    #plt.figure(figsize=[15,10])
    plt.imshow(Cloud2, interpolation='bilinear')
    plt.axis("off")
    #plt.title('Top words used in episode titles')
    st.pyplot(fig)

#############################################################################   


st.subheader('Top 5 Episodes:')
st.subheader('by Votes')
fig = go.Figure(data=[go.Table(header=dict(values=['Year of prod', 'Season','Episode Title', 'Votes', 'Director'], 
                                            line_color='darkslategray',
                                            fill_color='lightblue',
                                            align='left'),
                 cells=dict(values=[[2004, 1999, 1994, 2004, 1998], [10, 5, 1, 10, 4], ['The Last One: Part 2',
                            'The One Where Everybody Finds Out', "The One with Ross's Wedding", 'The One Where Monica Gets a Roommate:...',
                            'The One with the Proposal'], [10381, 8066,7560, 7440,7251], ['Kevin Bright', 'Michael Lembeck', 'James Burrows', 'Kevin Bright', 'Kevin Bright']])
                              )])
fig.update_layout(
    width=800,
    height=300,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
        )
    )

st.plotly_chart(fig)

######################################################


st.subheader('by Star Rating')
fig = go.Figure(data=[go.Table(header=dict(values=['Year of prod','Season','Episode Title', 'Stars', 'Director'], 
                                            line_color='darkslategray',
                                            fill_color='lightblue',
                                            align='left'),
                 cells=dict(values=[[2004, 1999, 2004, 1998,1996], [10,5,10,4,2],['The Last One: Part 2',
                            'The One Where Everybody Finds Out', 'The Last One: Part 1', 'The One with the Embryos',
                            'The One with the Prom Video'], [9.7, 9.7,9.5, 9.5,9.4],['Kevin Bright', 'Michael Lembeck', 
                            'Kevin Bright', 'Kevin Bright', 'James Burrows']])
                              )])
fig.update_layout(
    width=800,
    height=300,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
        )
    )

st.plotly_chart(fig)