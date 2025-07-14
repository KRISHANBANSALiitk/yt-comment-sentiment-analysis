# import mlflow
# from mlflow.tracking import MlflowClient

# # Set MLflow tracking URI if hosted remotely
# mlflow.set_tracking_uri("http://ec2-16-171-150-100.eu-north-1.compute.amazonaws.com:5000/")  # Replace with your server URI

# # Load model from the model registry
# def load_model_from_registry(model_name, model_version):
#     model_uri = f"models:/{model_name}/{model_version}"
#     model = mlflow.pyfunc.load_model(model_uri)
#     return model

# # Example usage
# model = load_model_from_registry("yt_chrome_plugin_model", "1")  # Replace with your model name and version
# print("Model loaded successfully!")

## Above code is just to test whether we are able to fecth model from model registry or not##


import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    # Set MLflow tracking URI to your server
    mlflow.set_tracking_uri("http://ec2-16-171-150-100.eu-north-1.compute.amazonaws.com:5000/")  # Replace with your MLflow tracking URI
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer
    return model, vectorizer

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl")  # Update paths and versions as needed

@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        #comments = [item['text'] for item in comments_data]
        #timestamps = [item['timestamp'] for item in comments_data]
        comments = comments_data

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform and convert to DataFrame
        transformed_sparse = vectorizer.transform(preprocessed_comments)
        feature_names = vectorizer.get_feature_names_out()
        transformed_comments = pd.DataFrame(transformed_sparse.toarray(), columns=feature_names)

        # Make predictions
        predictions = model.predict(transformed_comments).tolist()

        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)