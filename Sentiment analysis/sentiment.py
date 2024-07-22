import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import moviepy.editor as mp
import speech_recognition as sr
import joblib as jb

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function for text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters, punctuation, and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text into individual words or tokens
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Read the dataset
dataset_path = r"D:\Graduation\omar salem's project-20240429T152207Z-001\omar salem_s project\project\project\twitter_training.csv"  
try:
    data = pd.read_csv(dataset_path)
except FileNotFoundError:
    print("Dataset file not found.")
    exit()

# Replace missing values with an empty string
data['tweet'].fillna('', inplace=True)

# Preprocess the text data
data['Processed_Text'] = data['tweet'].apply(preprocess_text)

# Feature Engineering: Convert text data into numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data['Processed_Text'])
y = data['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Perform GridSearchCV for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the model with the best parameters
best_rf_classifier = RandomForestClassifier(**best_params)
best_rf_classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred = best_rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

accuracy_percentage = accuracy * 100
print("Accuracy:", accuracy_percentage, "%")

# Extract text from the video
video_path = r"D:\Graduation\omar salem's project-20240429T152207Z-001\omar salem_s project\project\project\New iphone.mp4"  # Replace with the path to your video file
try:
    video_clip = mp.VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile("temp_audio.wav", codec='pcm_s16le')
    video_clip.close()
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile("temp_audio.wav")
    with audio_file as source:
        audio_data = recognizer.record(source)
        video_text = recognizer.recognize_google(audio_data)
except Exception as e:
    print("Error extracting text from the video:", e)
    exit()

# Preprocess the extracted text
processed_video_text = preprocess_text(video_text)

# Convert the preprocessed text into numerical features using TF-IDF
X_video = tfidf_vectorizer.transform([processed_video_text])

# Predict sentiment of the video text
predicted_sentiment = best_rf_classifier.predict(X_video)
print("Predicted Sentiment of the Video Text:", predicted_sentiment[0])

# Define the directory path
directory = r"D:\Graduation\omar salem's project-20240429T152207Z-001\omar salem_s project"
# Ensure the directory exists, if not create it
if not os.path.exists(directory):
    os.makedirs(directory)

# Define the file path for saving the model
model_file_path = os.path.join(directory, 'sentiment_model.h5')
# Save the model to disk
jb.dump(best_rf_classifier, model_file_path)

print("Model saved successfully as", model_file_path)
# Load the saved model from disk
loaded_model = jb.load(model_file_path)


