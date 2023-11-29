# sentiment_model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_sentiment_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    return model, vectorizer

def predict_sentiment(model, vectorizer, X_test):
    X_test_vectorized = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vectorized)
    return predictions