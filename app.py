from flask import Flask, request, render_template
import joblib
import re
from bs4 import BeautifulSoup

app = Flask(__name__)

# Loading the model and vectorizer
model = joblib.load('/sentiment_model.pkl')
vectorizer = joblib.load('/tfidf_vectorizer.pkl')

# Cleaning function
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    clean_review = clean_text(review)
    review_vec = vectorizer.transform([clean_review])
    prediction = model.predict(review_vec)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
