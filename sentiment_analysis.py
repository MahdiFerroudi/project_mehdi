# sentiment_analysis.py
import sys
import nltk
from data_loader import load_data
from sentiment_model import predict_sentiment

# Spécifiez le chemin de téléchargement pour les ressources NLTK
nltk.data.path.append("/app/nltk_data")

def analyze_sentiment(file_path):
    data = load_data(file_path)
    prediction = predict_sentiment(data)
    return prediction

if __name__ == "__main__":
    # Assurez-vous que nltk.download fonctionne avec le bon chemin
    nltk.download("vader_lexicon", download_dir="/app/nltk_data")

    if len(sys.argv) != 2:
        print("Usage: python sentiment_analysis.py <path_to_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    result = analyze_sentiment(file_path)
    print("Sentiment Analysis Result:", result)