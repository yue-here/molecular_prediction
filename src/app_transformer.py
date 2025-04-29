from flask import Flask, request, jsonify
import mlflow
import pandas as pd
import numpy as np
from utils import build_vocabulary, smiles_tokenizer

app = Flask(__name__)

# Load a pytorch model from MLflow in pyfunc format
mlflow.set_tracking_uri("http://localhost:5000")
model_uri = 'runs:/f6ef772fad0346daacd081280392cc12/model_epoch_100'
model = mlflow.pyfunc.load_model(model_uri)

# Set up vocabulary
df = pd.read_csv("./data.csv")
specials = ["<PAD>", "pKi", "pIC50", "ENZ1", "ENZ2", "ENZ3", "ENZ4"]
vocab = build_vocabulary(df, specials)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract relevant fields from the data
        smiles = data['SMILES']
        measurement_type = data['measurement_type']
        enzyme_name = data['enzyme_name']

        # Tokenize the SMILES string
        tokens = smiles_tokenizer(smiles)
        token_ids = [vocab[token] for token in tokens]
        token_ids = [vocab[measurement_type], vocab[enzyme_name]] + token_ids

        # Convert to tensor
        input_tensor = np.array([token_ids])

        # Make predictions
        predictions = model.predict(input_tensor)
        
        # Return predictions as JSON
        return jsonify(predictions.tolist())
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)