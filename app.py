from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS 

# Initialize Flask app
app = Flask(__name__)

CORS(app)

# Load the pre-trained model
model = joblib.load('house_price_model.pkl')

# Initialize the scaler (you should use the same scaler as in the training)
scaler = StandardScaler()

@app.route('/')
def home():
    # Render the index.html file when the user visits the root URL
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json

    # Extract features from JSON
    area = data.get('area')
    bedrooms = data.get('bedrooms')
    bathrooms = data.get('bathrooms')
    stories = data.get('stories')
    mainroad = 1 if data.get('mainroad') == 'yes' else 0
    guestroom = 1 if data.get('guestroom') == 'yes' else 0
    basement = 1 if data.get('basement') == 'yes' else 0
    hotwaterheating = 1 if data.get('hotwaterheating') == 'yes' else 0
    airconditioning = 1 if data.get('airconditioning') == 'yes' else 0
    parking = data.get('parking')
    prefarea = 1 if data.get('prefarea') == 'yes' else 0
    furnished = 1 if data.get('furnishingstatus') == 'furnished' else 0
    semi_furnished = 1 if data.get('furnishingstatus') == 'semi-furnished' else 0
    unfurnished = 1 if data.get('furnishingstatus') == 'unfurnished' else 0

    # Combine features into a numpy array
    features = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating,
                          airconditioning, parking, prefarea, furnished, semi_furnished, unfurnished]])

    # Scale the input features (the same way as in the training)
    # Fit_transform should have been done before. Here we are just transforming as model is already trained
    features = scaler.fit_transform(features)

    # Predict using the loaded model
    prediction = model.predict(features)

    # Return the result as a JSON response
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
