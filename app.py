from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained model

model = joblib.load('house_price_model.pkl')  # Make sure to save your trained model to this file

# Initialize the StandardScaler (since you used it during training)
scaler = StandardScaler()
# You'll need to fit the scaler on the same data that was used during model training
# Here we provide a dummy fitting for illustration; replace this with your original training data.
scaler.fit(np.random.rand(10, 12))  # Dummy fitting. Replace with your original feature scaling

@app.route('/predict', methods=['POST'])
def predict():
    # Extract JSON data from request
    data = request.get_json(force=True)
    
    # Extract input features from JSON
    input_features = [
        data['area'],
        data['bedrooms'],
        data['bathrooms'],
        data['stories'],
        1 if data['mainroad'] == 'yes' else 0,
        1 if data['guestroom'] == 'yes' else 0,
        1 if data['basement'] == 'yes' else 0,
        1 if data['hotwaterheating'] == 'yes' else 0,
        1 if data['airconditioning'] == 'yes' else 0,
        data['parking'],
        1 if data['prefarea'] == 'yes' else 0,
        1 if data['furnishingstatus'] == 'furnished' else 0,
        1 if data['furnishingstatus'] == 'semi-furnished' else 0,
        1 if data['furnishingstatus'] == 'unfurnished' else 0
    ]

    # Convert the input to a numpy array and scale it
    input_features = np.array(input_features).reshape(1, -1)
    input_features_scaled = scaler.transform(input_features)

    # Predict the house price
    prediction = model.predict(input_features_scaled)
    
    # Return the result as JSON
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
