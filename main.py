# main.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_loader import load_movie_reviews_data
from sentiment_model import train_sentiment_model, predict_sentiment

if __name__ == "__main__":
    reviews, labels = load_movie_reviews_data()
    X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)
    trained_model, trained_vectorizer = train_sentiment_model(X_train, y_train)
    predictions = predict_sentiment(trained_model, trained_vectorizer, X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")