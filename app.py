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
    try:
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
        furnishingstatus = data.get('furnishingstatus')
        furnished = 1 if furnishingstatus == 'furnished' else 0
        semi_furnished = 1 if furnishingstatus == 'semi-furnished' else 0
        unfurnished = 1 if furnishingstatus == 'unfurnished' else 0
        
        # Check for missing data
        if None in [area, bedrooms, bathrooms, stories, parking]:
            return jsonify({'error': 'Missing or invalid data'}), 400
        
        # Check for correct data types
        if not all(isinstance(x, (int, float)) for x in [area, bedrooms, bathrooms, stories, parking]) or \
           not isinstance(mainroad, int) or not isinstance(guestroom, int) or not isinstance(basement, int) or \
           not isinstance(hotwaterheating, int) or not isinstance(airconditioning, int) or not isinstance(prefarea, int) or \
           not isinstance(furnished, int) or not isinstance(semi_furnished, int) or not isinstance(unfurnished, int):
            return jsonify({'error': 'Invalid data types'}), 400
        
        # Combine features into a numpy array
        features = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating,
                              airconditioning, parking, prefarea, furnished, semi_furnished, unfurnished]])
        
        # Scale the input features (the same way as in the training)
        features = scaler.fit_transform(features)
        
        # Predict using the loaded model
        prediction = model.predict(features)
        
        # Return the result as a JSON response
        return jsonify({'predicted_price': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000, debug=True)
