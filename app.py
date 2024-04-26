from flask import Flask, request, jsonify
import pickle
import json
import pandas as pd

app = Flask(__name__)

# Load the model and feature data
with open('new_rf_model.pickle', 'rb') as model_file:
    model = pickle.load(model_file)

with open('features_data.json', 'r') as feature_file:
    features_data = json.load(feature_file)
    feature_columns = features_data['columns']
class_labels=['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
       'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
       'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
       'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

# Route to accept input and return predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Validate and extract input data
    try:
        test_data = [data[feature] for feature in feature_columns]
    except KeyError as e:
        return jsonify({"error": f"Missing data for {str(e)}"}), 400

    # Create a DataFrame for the model input
    input_data = pd.DataFrame([test_data], columns=feature_columns)

    # Predict the crop
    prediction = model.predict(input_data)
    if prediction[0] >= len(class_labels):
        return jsonify({"error": "Prediction index out of range"}), 500
    predicted_crop = class_labels[prediction[0]]

    # Return the prediction
    return jsonify({"predicted_crop": predicted_crop})


if __name__ == '__main__':
    app.run(debug=True)
