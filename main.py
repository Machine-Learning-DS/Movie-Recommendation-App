import pandas as pd
import numpy as np
import nltk
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#RAW DATA->TOKENIZATION->TEXT CLEANING->VECTORIZATION->ML ALGO
#RAW DATA->DATASET  WE EXTRACT DATA
#TOKENIZATION ->COONVERT PASSSAGE TO WORDS/SNTENCES(MEANINGFUL FORM)
#TEXT CLEANING -> INCLUDES STANDARDIZATION,STEMMING/LEMMITIZATION,REMOVING PUNCTUATIONS AND STOP WORDS,CONVERTING TO LOWERCASE,URLS,ETC.
#VECTORIZATION ->CONVERTING TEXTUAL MEANINGFUL DATA INTO NUMERICAL VECTORS THAT ML ALGOS CAN 
# ANALYZE EFFICIENTLY TYPES(depending on use-case ex spam detection,semantics understanding,tagging,sentiment analaysis,language translation) : 1)BAG OF WORDS(WHAT WE DID) 2) GloVe(global vectors) 3) Word Embedding 4)TF-IDF


# # TAKING THE DATA AS INPUT







credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')
# print(movies.info())
# print (credits.info())







# # FILTERING OUT THE DATA WE NEED







# # columns selected to consider them in tags:
# # genre id(used when creating website not for tags though)  keywords title overview popularity cast crew
movies = movies[['title','overview','genres','keywords','popularity']]
credits = credits[['movie_id','title','cast','crew']]
movies = movies.merge(credits,on='title')
# print(movies.info())
# print(movies.shape)








# # PREPROCESSING THE DATA






# # 1)Filtering the missing data:

# print(movies.isnull().sum())
# only overview has 3 missing data but since it essential we'll 
# be removing those movies which dont have an overview
movies.dropna(inplace=True)
# print(movies.isnull().sum())


# # 2)Filtering the duplicate data:
#there wasn't any duplicate data in our case
# print(movies.duplicated().sum())



# # 3)Converting data in correct format:

# our list of data is actually a string and actually the indices 
# must be integers for our function to work and hence
# inorder to convert this string of list to list there exist this module
# ast.literal_eval()


def formatter(obj,type):
    Vals=[]
    for i in ast.literal_eval(obj):
        name = i['name'].replace(" ","").lower() #removing the spaces and converting them to lowercase
        if type == 1:
            Vals.append(name)
        elif type==2:
            if len(Vals)==3:
                break
            Vals.append(name)
        elif type==3:
            if i['job']=='Director':
                Vals.append(name)
                break
    return Vals

movies['keywords'] = movies['keywords'].apply(formatter, args=(1,))
movies['genres'] = movies['genres'].apply(formatter, args=(1,))
# just taking the first 3 important cast members only into consideration
movies['cast'] = movies['cast'].apply(formatter, args=(2,))
# just taking director name into consideration
movies['crew'] = movies['crew'].apply(formatter, args=(3,))
# converting the overview passage into a list of words lowercased
movies['overview'] = movies['overview'].apply(lambda i: [word.lower() for word in i.split()])
# converting populrity to a list as well so that its easier to append in the future
movies['popularity'] = movies['popularity'].apply(lambda x: [x])
# print(movies['keywords'][0])
# print(movies['genres'][0])
# print(movies['cast'][0])
# print(movies['overview'][0])
# print(movies['popularity'][0])


# # 4)Creating the tags column by appending the rest of the columns and removing the rest

movies['tags']=movies['overview']+movies['cast']+movies['crew']+movies['genres']+movies['keywords']+movies['popularity']
# print(movies['tags'][0])
# converting tags back to a passage form for easier comparison
movies['tags'] = movies['tags'].apply(lambda i: " ".join(map(str, i)))
# our final preprocessed data!
finaldata = movies[['movie_id','title','tags']]
# print(finaldata['tags'][0])

# # 5) Data Cleaning
#stemming functions/classes used ->nltk.stem.porter ->porterstemmer
ps = PorterStemmer()
# print(ps.stem('dance'))
def stem(text):
    stemmed_list=[]
    for i in text.split():
        stemmed_list.append(ps.stem(i))
    return " ".join(stemmed_list) #converting it back to string

finaldata['tags'] = finaldata['tags'].apply(stem)
# print(finaldata['tags'][0])
#functions/classes used ->countvectorization

#removing stop words and selecting top 5000 most frequent words
cv = CountVectorizer(max_features=5000,stop_words='english',)
vectors = cv.fit_transform(finaldata['tags']).toarray()#we explicitly convetted it inot a numpy arry since by default it would have returned a sparse matrix
# print(vectors)
# print(cv.get_feature_names_out())

#instead of taking euclidian distance taking cosine distance is more preffered for high dimensional data space(in our case 4806)
#function/classes used -> cosine similarity
cosine_dist =cosine_similarity(vectors)
# print(cosine_dist.shape)
# print(sorted_dist[1])
# # FINALLY FUNCTION TO RECOMMEND THE MOVIE!

def recommend(movie):
    idx  =finaldata[finaldata['title']==movie].index[0]
    distances= cosine_dist[idx]
    sorted_dist = sorted(list(enumerate(distances)),reverse=True,key=lambda i:i[1])[1:6]
    for i in sorted_dist:
        print(finaldata.iloc[i[0]].title)
    return 
recommend('Avatar')
 















# def preprocess_text(text):
#     # Remove punctuation
#     text = re.sub(r'[^\w\s]', '', text)

#     # Tokenize the text
#     words = nltk.word_tokenize(text)

#     # Remove stop words
#     stop_words = set(stopwords.words('english'))
#     words = [word for word in words if word.lower() not in stop_words]

#     # Lemmatize the words
#     lemmatizer = WordNetLemmatizer()
#     words = [lemmatizer.lemmatize(word) for word in words]

#     return ' '.join(words)

# # Apply the preprocessing function
# finaldata['tags'] = finaldata['tags'].apply(preprocess_text)

# # Print the preprocessed data
# print(finaldata['tags'][0])







# # VECTORIZATION







