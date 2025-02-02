"""
Misinformation Detection and Fact-Checking Prototype
------------------------------------------------------
Steps: 3. Flask App Setup and 4. App Execution
"""

import joblib
from flask import Flask, request, render_template

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

test_accuracy = 0.67

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html').format(test_accuracy * 100)

@app.route('/check', methods=['POST'])
def check_misinformation():
    user_text = request.form.get('user_text', '')
    user_vector = vectorizer.transform([user_text])
    
    prediction = model.predict(user_vector)[0]
    
    if prediction == 1:
        fact_check_message = "This text is flagged as potential misinformation. Please verify with official sources."
    else:
        fact_check_message = "This text appears to be accurate."
    
    return f'''
    <html>
    <head>
        <title>Result</title>
    </head>
    <body>
        <h2>Analysis Result</h2>
        <p>Your text: <em>{user_text}</em></p>
        <p><strong>Outcome: {fact_check_message}</strong></p>
        <a href="/">Go Back</a>
    </body>
    </html>
    '''

if __name__ == "__main__":
    app.run(debug=True)