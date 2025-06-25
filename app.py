from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('spam_classifier.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message'].lower()  # Convert to lowercase for easier checking

    # ✅ Custom spam keywords that you want to block
    flagged_phrases = ["trust me", "i love you", "honey", "baby", "swear to god"]

    # ✅ Manual override: If any of these keywords are found, flag as spam
    if any(phrase in message for phrase in flagged_phrases):
        prediction = "spam"
    else:
        prediction = model.predict([message])[0]

    return render_template('index.html', prediction=prediction, message=message)


if __name__ == '__main__':
    app.run(debug=True)
