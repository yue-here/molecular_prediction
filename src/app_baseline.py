from flask import Flask, request, jsonify
import mlflow
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator


app = Flask(__name__)

# Load the model from MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model_uri = 'runs:/8ac20f5615da41f0852b29fcab2784c1/gradient_boosting_model'
model = mlflow.pyfunc.load_model(model_uri)

# Create preprocessor for input
def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        calc = MolecularDescriptorCalculator([desc[0] for desc in Descriptors.descList])
        features = calc.CalcDescriptors(mol)
    else:
        features = {desc[0]: None for desc in Descriptors.descList}
    return features

# Load main dataset as reference
df = pd.read_csv("./data.csv")

# Create a one-hot encoder for measurement_type and enzyme_name
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder = encoder.fit(df[['measurement_type', 'enzyme_name']])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Data should be in the format SMILES; measurement_type; enzyme_name
        data_df = pd.DataFrame(data, columns=['SMILES', 'measurement_type', 'enzyme_name'])
        encoded = encoder.transform(data_df[['measurement_type', 'enzyme_name']])
        features = data_df['SMILES'].apply(featurize_smiles).tolist()
        input_df = pd.concat([pd.DataFrame(encoded), pd.DataFrame(features)], axis=1)

        # Make predictions
        predictions = model.predict(input_df)
        
        # Return predictions as JSON
        return jsonify(predictions.tolist())
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)