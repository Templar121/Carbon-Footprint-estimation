from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from CarbonFootprint.pipeline.prediction import PredictionPipeline
import os

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",
            "https://carbonfootprintemission.netlify.app"
        ]
    }
})  # Enable CORS for said origins

# Mappings for categorical features
# encoding_maps = {
#     'Body Type': {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3},
#     'Sex': {'male': 0, 'female': 1},
#     'Diet': {'vegan': 0, 'vegetarian': 1, 'omnivore': 2, 'pescatarian': 3},
#     'How Often Shower': {'daily': 0, 'twice a day': 1, 'more frequently': 2, 'less frequently': 3},
#     'Heating Energy Source': {'electricity': 0, 'natural gas': 1, 'coal': 2, 'wood': 3},
#     'Transport': {'public': 0, 'walk/bicycle': 1, 'private': 2},
#     'Social Activity': {'never': 0, 'often': 1, 'sometimes': 2},
#     'Frequency of Traveling by Air': {'never': 0, 'rarely': 1, 'very frequently': 2, 'frequently': 3},
#     'Waste Bag Size': {'small': 0, 'medium': 1, 'large': 2, 'extra large': 3},
#     'Energy efficiency': {'Sometimes': 0, 'Yes': 1, 'No': 2}
# }

input_columns = [
    "Body Type", "Sex", "Diet", "How Often Shower", "Heating Energy Source", 
    "Transport", "Social Activity", "Monthly Grocery Bill", "Frequency of Traveling by Air", 
    "Vehicle Monthly Distance Km", "Waste Bag Size", "Waste Bag Weekly Count", 
    "How Long TV PC Daily Hour", "How Many New Clothes Monthly", 
    "How Long Internet Daily Hour", "Energy efficiency"
]

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'success', 'message': 'Carbon Emission API is live.'}), 200

@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return jsonify({
        'status': 'success',
        'message': 'Model training completed successfully.'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_data = []
        for col in input_columns:
            if col not in data:
                return jsonify({'status': 'error', 'message': f'Missing input: {col}'}), 400
            try:
                input_data.append(float(data[col]))
            except ValueError:
                return jsonify({'status': 'error', 'message': f'Invalid numeric value for {col}: {data[col]}'}), 400

        input_array = np.array(input_data).reshape(1, -1)
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(input_array)

        return jsonify({
            'status': 'success',
            'CarbonEmission': float(prediction[0])
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
