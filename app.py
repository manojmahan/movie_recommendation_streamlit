import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from datetime import timedelta
from ast import literal_eval
import bs4 as bs
import urllib.request
import pickle
from urllib.request import Request, urlopen

import base64
import math


millnames = ['',' Thousand',' Million',' Billion',' Trillion']
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("transform.pkl","rb"))

def highlight_survived(s):
    return ['background-color: red']*len(s)

def millify(n):
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


st.set_page_config(layout='wide')
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('image.png')  

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=6191299c63f9897b9302669be6ecc18b&language=en-US"
    data = requests.get(url)
    data = data.json()
    #print(data)
    full_path = "https://image.tmdb.org/t/p/w500" + (data['poster_path'] or "")
    return full_path

def get_video_link(movie_id):
    
    url = f"http://api.themoviedb.org/3/movie/{movie_id}/videos?api_key=6191299c63f9897b9302669be6ecc18b"
    video_data = requests.get(url)
    video_data = video_data.json()
    #print("video_data",video_data)
    try :
        for i in video_data["results"]:
            if i["type"]=='Trailer':
                key=i['key']
        full_video_path = "https://www.youtube.com/watch?v=" + key
    except :
        full_video_path="https://www.youtube.com/"
    #print("video_data",video_data)
    #print("key",key)
    #print("full_video_path",full_video_path)
    return full_video_path

def create_similarity():
    dataset = pd.read_csv('final_dataset.csv')
    # creating a count matrix
    countvec = CountVectorizer(stop_words="english")
    count_matrix = countvec.fit_transform(dataset["soup"])
    # creating a similarity score matrix
    cosin_sim = cosine_similarity(count_matrix,count_matrix)
    return cosin_sim


def recommend(movie):
    indices = pd.Series(dataset.index,index=dataset["title_x"])
    ids = indices[movie]
    cosin_sim = create_similarity()
    x = sorted(list(enumerate(cosin_sim[ids])),key=lambda x:x[1],reverse=True)[0:11]
    lst = [i[0] for i in x]
    recommended_movie_names = []
    recommended_movie_posters = []
    recommended_movie_video = []
    for i in lst[1:]:
        movie_id = dataset.iloc[i].id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_video.append(get_video_link(movie_id))
        recommended_movie_names.append(dataset.iloc[i].title_x)
    return recommended_movie_names,recommended_movie_posters,recommended_movie_video

def recommend_genre(selected_genre):
    dataset = pd.read_csv("final_dataset.csv")
    dataset = dataset[dataset["genres"].str.contains(selected_genre)]
    dataset = dataset.drop(["index"],axis=1)
    top_10 = (dataset.sort_values("weighted_rating",ascending=False)[0:10]).copy()
    recommended_movie_names_genre = []
    recommended_movie_posters_genre = []
    recommended_movie_video_genre = []
    for i in range(10):
        movie_id = top_10.iloc[i].id
        
        # fetch poster from api
        recommended_movie_posters_genre.append(fetch_poster(movie_id))
        recommended_movie_names_genre.append( top_10.iloc[i].title_x)
        recommended_movie_video_genre.append(get_video_link(movie_id))
    return recommended_movie_names_genre,recommended_movie_posters_genre,recommended_movie_video_genre

def sentiment_df(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/external_ids?api_key=6191299c63f9897b9302669be6ecc18b"
    data = requests.get(url)
    data = data.json()
    imdb_id  =data["imdb_id"]
    url = f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt'

# add a fake browser header
    req = Request(
        url,
    headers={"User-Agent": "Mozilla/5.0"}
    )

    sauce = urlopen(req).read()
    soup1 = bs.BeautifulSoup(sauce,'lxml')
    soup_result = soup1.find_all("div",{"class":"text show-more__control"})
    reviews_list = [] # list of reviews
    reviews_status = [] # list of comments (good or bad)
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # passing the review to our model
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = model.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')
    reviews_dataframe = pd.DataFrame(reviews_list,columns=["reviews"])
    reviews_dataframe["Sentiments"] = reviews_status
    if reviews_dataframe.shape[0]>10 :
        reviews_dataframe= reviews_dataframe[0:10]
    return reviews_dataframe


st.markdown(f'''<h1 style="color:white;">Movie Recommender System</h1>''',unsafe_allow_html=True)
dataset = pd.read_csv("final_dataset.csv")
movie_list = dataset['title_x'].values
st.markdown(f'''<h1 style="font-size:1rem; color: #FFFFFF">Type or select a movie from the dropdown</h1>''',unsafe_allow_html=True)
selected_movie = st.selectbox('_',
    movie_list)
features = ["genres"]
for feature in features:
    dataset[feature]= dataset[feature].apply(literal_eval)
d = dict()
for i in list(dataset["genres"]):
    for j in i:
        if j not in d:
            d[j]=j
genres_list = d.values()





if st.button('Show Recommendation according to movie'):
    
    recommended_movie_names,recommended_movie_posters,recommended_movie_video = recommend(selected_movie)
    #print("recommended_movie_video:",recommended_movie_video)
    indices = pd.Series(dataset.index,index=dataset["title_x"])
    ids = indices[selected_movie]
    cosin_sim = create_similarity()
    x = sorted(list(enumerate(cosin_sim[ids])),key=lambda x:x[1],reverse=True)[0]
    selected_movie_id= dataset.iloc[x[0]].id
    url = f"https://api.themoviedb.org/3/movie/{selected_movie_id}?api_key=6191299c63f9897b9302669be6ecc18b&language=en-US"
    data_select = requests.get(url)
    data_select = data_select.json()
    
    full_path_select = "https://image.tmdb.org/t/p/w500" + data_select['poster_path']
    full_video = get_video_link(selected_movie_id)
    col1111, mid, col1112 = st.columns([5,1,20])
    with col1111:
        st.markdown(f'''
                        <a href={full_video}>
                            <img src={full_path_select} width="280" />
                        </a>''',
                        unsafe_allow_html=True
                    )
    with col1112:
        st.markdown(f'''<h1 style="font-size:1rem; color: #FFFFFF">Budget:  {millify(data_select["budget"])} USD</h1>''',unsafe_allow_html=True)
        st.markdown(f'''<h1 style="font-size:1rem; color: #FFFFFF">Revenue:  {millify(data_select["revenue"])} USD</h1>''',unsafe_allow_html=True)
        st.markdown(f'''<h1 style="font-size:1rem; color: #FFFFFF">Profit Percentage:  {round((data_select["revenue"]-data_select["budget"])/(data_select["budget"]+0.0000002),2)*100}%</h1>''',unsafe_allow_html=True)
        st.markdown(f'''<h1 style="font-size:1rem; color: #FFFFFF">Total Runtime:  {str(timedelta(minutes=data_select["runtime"]))[0]} Hour(s) {str(timedelta(minutes=data_select["runtime"]))[2:-3]} min(s) </h1>''',unsafe_allow_html=True)
        st.markdown(f'''<p> <h1 style="font-size:1rem; color: #FFFFFF">Average Rating:  {round(data_select["vote_average"],1)} &#9733; ({data_select["vote_count"]}votes)</h1> </p>''',unsafe_allow_html=True)
        st.markdown(f'''<h1 style="font-size:1rem; color: #FFFFFF">Overview:     {data_select["overview"]}</h1>''',unsafe_allow_html=True)
        # st.write("Budget:",millify(data_select["budget"]))
        # st.write("Revenue:",millify(data_select["revenue"]))
        # st.write("Profit Percentage:",round((data_select["revenue"]-data_select["budget"])/(data_select["budget"]),2),"%")
        # st.write("Total Runtime:",str(timedelta(minutes=data_select["runtime"]))[0],"Hour(s)",str(timedelta(minutes=data_select["runtime"]))[2:-3],"min(s)") 
        # st.write("Average Rating:",data_select["vote_average"],"(",data_select["vote_count"]," votes)") 
        # st.write("Overview:",data_select["overview"]) 
    
    
    "---"
    st.markdown(f'''<h1 style="font-size:3rem; color: #FFFFFF">Top five related movies</h1>''',unsafe_allow_html=True)
    st.markdown(f'''<h1 style="font-size:1rem; color: #FFFFFF">Click on any poster to watch the trailer</h1>''',unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'''
                        <a href={recommended_movie_video[0]}>
                            <img src={recommended_movie_posters[0]}  width="250" />
                            <figcaption style="color:white;">{recommended_movie_names[0]}</figcaption>
                        </a>''',
                        unsafe_allow_html=True
                    )
    with col2:
        st.markdown(f'''
                        <a href={recommended_movie_video[1]}>
                            <img src={recommended_movie_posters[1]} width="250"/>
                            <figcaption style="color:white;">{recommended_movie_names[1]}</figcaption>
                        </a>''',
                        unsafe_allow_html=True
                    )
        
    with col3:
        st.markdown(f'''
                        <a href={recommended_movie_video[2]}>
                            <img src={recommended_movie_posters[2]} width="250"/>
                            <figcaption style="color:white;">{recommended_movie_names[2]}</figcaption>
                        </a>''',
                        unsafe_allow_html=True
                    )
    with col4:
        st.markdown(f'''
                        <a href={recommended_movie_video[3]}>
                            <img src={recommended_movie_posters[3]} width="250" />
                            <figcaption style="color:white;">{recommended_movie_names[3]}</figcaption>
                        </a>''',
                        unsafe_allow_html=True
                    )
    with col5:
        st.markdown(f'''
                        <a href={recommended_movie_video[4]}>
                            <img src={recommended_movie_posters[4]} width="250" />
                            <figcaption style="color:white;">{recommended_movie_names[4]}</figcaption>
                        </a>''',
                        unsafe_allow_html=True
                    )
    sentiment_dataframe  = sentiment_df(selected_movie_id)
    dataframe_html = sentiment_dataframe.to_html()
    st.markdown(f'''<h1 style="font-size:3rem; color: #FFFFFF";>Sentiment Analysis on User Reviews</h1>''',unsafe_allow_html=True)
    st.markdown(f'''<h1 style="font-size:0.8rem; color: #FFFFFF" <body style="background-color:powderblue;"> >{dataframe_html} </body> </h1>''',unsafe_allow_html=True)
    "---"
st.markdown(f'''<h1 style="font-size:1rem; color: #FFFFFF">Type or select a genre from the dropdown</h1>''',unsafe_allow_html=True)
selected_genre = st.selectbox(
    " ",
    genres_list
)

if st.button('Show Recommendation According to genre'):
    recommended_movie_names,recommended_movie_posters,recommended_movie_video_genre = recommend_genre(selected_genre)
    "---"
    st.markdown(f'''<h1 style="font-size:3rem; color: #FFFFFF">Top five {selected_genre} movies</h1>''',unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'''
                        <a href={recommended_movie_video_genre[0]}>
                            <img src={recommended_movie_posters[0]} width="250" />
                            <figcaption style="color:white;">{recommended_movie_names[0]}</figcaption>
                        </a>''',
                        unsafe_allow_html=True
                    )
    with col2:
        st.markdown(f'''
                        <a href={recommended_movie_video_genre[1]}>
                            <img src={recommended_movie_posters[1]} width="250" />
                            <figcaption style="color:white;">{recommended_movie_names[1]}</figcaption>
                        </a>''',
                        unsafe_allow_html=True
                    )
    with col3:
        st.markdown(f'''
                        <a href={recommended_movie_video_genre[2]}>
                            <img src={recommended_movie_posters[2]} width="250" />
                            <figcaption style="color:white;">{recommended_movie_names[2]}</figcaption>
                        </a>''',
                        unsafe_allow_html=True
                    )
    with col4:
        st.markdown(f'''
                        <a href={recommended_movie_video_genre[3]}>
                            <img src={recommended_movie_posters[3]} width="250" />
                            <figcaption style="color:white;">{recommended_movie_names[3]}</figcaption>
                        </a>''',
                        unsafe_allow_html=True
                    )
    with col5:
        st.markdown(f'''
                        <a href={recommended_movie_video_genre[4]}>
                            <img src={recommended_movie_posters[4]} width="250" />
                            <figcaption style="color:white;">{recommended_movie_names[4]}</figcaption>
                        </a>''',
                        unsafe_allow_html=True
                    )


