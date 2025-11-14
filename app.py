from flask import Flask, request, jsonify, render_template
from joblib import load
from preprocess import clean_text


app = Flask(__name__)
vectorizer = load('models/tfidf_vectorizer.joblib')
classifier = load('models/classifier.joblib')


@app.route('/')
def index():
return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
data = request.json
text = data.get('text', '')
text_clean = clean_text(text)
x_tfidf = vectorizer.transform([text_clean])
pred = classifier.predict(x_tfidf)[0]
proba = classifier.predict_proba(x_tfidf).tolist()[0]
label_map = {1: 'Real', 0: 'Fake'}
return jsonify({'label': int(pred), 'label_name': label_map.get(int(pred)), 'probability': proba})


if __name__ == '__main__':
app.run(debug=True, host='0.0.0.0', port=5000)
