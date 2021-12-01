#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: danielsmetana7@ MABA CLASS
"""

import streamlit as st
import pandas as pd
import spacy
from string import punctuation
nlp = spacy.load("en_core_web_sm")
from sentence_transformers import SentenceTransformer
import scipy.spatial
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import re
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

st.title("Daniel Smetana Chicago Hotel Contextual Search Engine")

################################################################################

embedder = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv("hotelReviewsInChicago.csv")

df['hotelName'].drop_duplicates()

s = "."

df_combined = df.sort_values(['hotelName']).groupby('hotelName', sort=False).review.apply(s.join).reset_index(name='all_review')


df_combined['all_review'] = df_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','. ',x))

def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

df_combined['all_review']= df_combined['all_review'].apply(lambda x: lower_case(x))

df = df_combined

df_sentences = df_combined.set_index("all_review")

df_sentences = df_sentences["hotelName"].to_dict()
df_sentences_list = list(df_sentences.keys())


df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]


search_criteria1 = st.text_input("Please enter first search criteria: ")
search_criteria2 = st.text_input("Please enter second search criteria: ")

queries = [str(search_criteria1), str(search_criteria2)]

# Corpus with example sentences
corpus = df_sentences_list
corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)

model = SentenceTransformer('all-MiniLM-L6-v2')


paraphrases = util.paraphrase_mining(model, corpus)
query_embeddings_p =  util.paraphrase_mining(model, queries, show_progress_bar=True)


query_embeddings = embedder.encode(queries,show_progress_bar=True)




# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 5
st.text("Top 5 most similar hotels:")
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    st.write("\n\n=========================================================")
    st.write("==========================Query==============================")
    st.write("===",query,"=====")
    st.write("=========================================================")


    for idx, distance in results[0:closest_n]:
        row_dict = df.loc[df['all_review']== corpus[idx]]
        st.write("Hotel Name:  " , row_dict['hotelName'] , "\n")
        st.write("Score:   ", "%.4f" % (1-distance) , "\n" )
        #st.write(corpus[idx].strip())
        st.write("Reviews Summary:   ", summarize(corpus[idx].strip(), word_count = 100, split = False), "\n" )
        st.write("Reviews Top-25 Keywords:   ", keywords(corpus[idx].strip(), words = 25, lemmatize = True), "\n")
        # print("Title:  " , row_dict["title"][corpus[idx]] , "\n")
        # print("Abstract:  " , row_dict["abstract"][corpus[idx]] , "\n")
        #st.write("Abstract_Summary:  " , row_dict["abstract_summary"][corpus[idx]] , "\n")
        st.write("-------------------------------------------")
