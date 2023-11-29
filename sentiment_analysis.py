import sys
from data_loader import load_data
from sentiment_model import predict_sentiment

def analyze_sentiment(file_path):
    data = load_data(file_path)
    prediction = predict_sentiment(data)
    return prediction

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sentiment_analysis.py <path_to_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    result = analyze_sentiment(file_path)
    print("Sentiment Analysis Result:", result)
