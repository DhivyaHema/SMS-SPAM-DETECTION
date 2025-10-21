from flask import Flask, request, render_template
import joblib
from src.preprocessing import clean_text

app = Flask(__name__)
model = joblib.load('models/spam_classifier.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    if request.method == 'POST':
        msg = request.form['message']
        cleaned = clean_text(msg)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
