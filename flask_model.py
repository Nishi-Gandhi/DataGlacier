# app.py - Flask API
from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('titanic_model.pkl', 'rb'))

# New route for the homepage
@app.route('/')
def home():
    return render_template_string("<h1>You are using the Flask App</h1>")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    prediction = model.predict(np.array(data).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='localhost')