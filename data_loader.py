# data_loader.py
from nltk.corpus import movie_reviews
import nltk

nltk.download('movie_reviews')

def load_movie_reviews_data():
    positive_reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids('pos')]
    negative_reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids('neg')]
    labels = ['pos'] * len(positive_reviews) + ['neg'] * len(negative_reviews)
    reviews = positive_reviews + negative_reviews
    return reviews, labels
