#importing libraries
import streamlit as st
# import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load
import urllib
from PIL import Image

amazon_data_embeddings = load(open('./tfidf_vectors/amazon_data_embeddings.pkl', 'rb'))
flipkart_data_embeddings = load(open('./tfidf_vectors/flipkart_data_embedd.pkl', 'rb'))
input_vectorizer  = load(open('./tfidf_vectors/input_vecto.pkl', 'rb'))

df_a = pd.read_csv('./data/amazon_process_data.csv')
df_f = pd.read_csv('./data/flipkart_process_data.csv')


# function to preprocess text
def preprocess(text):
    stemmer = PorterStemmer()
    text = ' '.join(stemmer.stem(word) for word in text.split())
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()  # consider only alphabets
    text = ' '.join(text.split()) # to remove spaces between the text
    return text


# Header
st.header("Similar product comparison")


prod_name = st.text_input("Enter Your product name")


# Create a button, that when clicked, shows a result
if(st.button("Search")):

    # prod_name = input('Enter product name: ')
    prod_name_processed = preprocess(prod_name)

    # convert the text into vector embedding
    embed = input_vectorizer.transform([prod_name_processed])

    res_f = np.dot(embed, flipkart_data_embeddings.T)

# select the index which has highest similarity
    idx_f = np.argmax(res_f) 

# display the most similar product from amazon dataset
    product_name_f, retail_price_f, discounted_price_f = df_f.loc[idx_f, ['product_name', 'retail_price', 'discounted_price']]
    retail_price_f = int(retail_price_f)
    discounted_price_f = int(discounted_price_f)
    flipkart = [product_name_f, retail_price_f, discounted_price_f]
    print(f'flipkart: {idx_f} {product_name_f}')

    # find the cosine similarity of our input sent with all sentence embedd of amazon
    res_a = np.dot(embed, amazon_data_embeddings.T)

    # select the index which has highest similarity
    idx_a = np.argmax(res_a) 

    # display the most similar product from amazon dataset
    product_name_a, retail_price_a, discounted_price_a = df_a.loc[idx_a, ['product_name', 'retail_price', 'discounted_price']]
    print(f'amazon: {idx_a} {product_name_a}')
    
    amazon = [product_name_a, retail_price_a, discounted_price_a]
    prod_details = flipkart + amazon
    cols = ['Product name in Flipkart', 'Retail Price in Flipkart', 'Discounted Price in Flipkart', 'Product name in Amazon', 'Retail Price in Amazon', 'Discounted Price in Amazon']
    data = {k:v for k, v in zip(cols, prod_details)}
    df = pd.DataFrame(data=data, index=[0])
 
    st.table(df)

