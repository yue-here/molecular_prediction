import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from transformer.dataset import (
    create_stratified_splits
)

if __name__ == "__main__":
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Gradient boosting test loss")

    # Load your main dataset
df = pd.read_csv("./data.csv")

    # Create splits (same as transformer version to maintain same test set)
    train_df, dev_df, test_df = create_stratified_splits(
        df,
        measurement_col='measurement_type',
        enzyme_col='enzyme_name',
        test_per_class=60,
        dev_per_class=60,
        pki_only=False,
        pki_only_dev_test=True
    )

    feature_df = pd.read_csv('./enzyme_dataset_features.csv')

    # Merge on index after splits to keep the same test set
    train_df = pd.merge(train_df, feature_df, left_index=True, right_index=True, how='left')
    dev_df = pd.merge(dev_df, feature_df, left_index=True, right_index=True, how='left')
    test_df = pd.merge(test_df, feature_df, left_index=True, right_index=True, how='left')

    # One-hot encode measurement_type and enzyme_name
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    train_encoded = encoder.fit_transform(train_df[['measurement_type', 'enzyme_name']])
    dev_encoded = encoder.transform(dev_df[['measurement_type', 'enzyme_name']])
    test_encoded = encoder.transform(test_df[['measurement_type', 'enzyme_name']])

    # Get the column names for the encoded features
    encoded_columns = encoder.get_feature_names_out(['measurement_type', 'enzyme_name'])

    # Append new columns to the data
    train_features = pd.DataFrame(train_encoded, columns=encoded_columns)
    dev_features = pd.DataFrame(dev_encoded, columns=encoded_columns)
    test_features = pd.DataFrame(test_encoded, columns=encoded_columns)

    # Add any numeric columns from feature_df (besides measurement_value)
    extra_cols = [c for c in feature_df.columns if c not in ['SMILES','measurement_type','enzyme_name','measurement_value']]
    train_features = pd.concat([train_features, train_df[extra_cols].reset_index(drop=True)], axis=1)
    dev_features = pd.concat([dev_features, dev_df[extra_cols].reset_index(drop=True)], axis=1)
    test_features = pd.concat([test_features, test_df[extra_cols].reset_index(drop=True)], axis=1)

    # Prepare targets
    train_y = train_df['measurement_value']
    dev_y = dev_df['measurement_value']
    test_y = test_df['measurement_value']

    # Load model as a PyFuncModel.
    model = 'runs:/8ac20f5615da41f0852b29fcab2784c1/gradient_boosting_model'
    model = mlflow.pyfunc.load_model(model)
    # model.predict(test_features)

    # Run model with test loader and calculate loss
    test_preds = model.predict(test_features)

    test_loss = mean_squared_error(test_y, test_preds)
    mlflow.log_metric("test_loss", test_loss)

    # Build a DataFrame with the test results
    test_results = pd.DataFrame({
        'measurement_type': test_df['measurement_type'],
        'enzyme_name': test_df['enzyme_name'],
        'y_true': test_y,
        'y_pred': test_preds
    })
    grouped = test_results.groupby(['measurement_type', 'enzyme_name'])

    rows = []
    for (mtype, enzyme), group_df in grouped:
        group_loss = mean_squared_error(group_df['y_true'], group_df['y_pred'])
        mlflow.log_metric(f"test_loss_{mtype}_{enzyme}", group_loss)
        rows.append({'measurement_type': mtype, 'enzyme_name': enzyme, 'group_loss': group_loss})

    results_df = pd.DataFrame(rows)

    # # Log the grouped losses as a CSV artifact in MLflow
    # with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    #     results_df.to_csv(f.name, index=False)
    #     mlflow.log_artifact(f.name, artifact_path="test_loss_by_measurement_enzyme")
    #     os.remove(f.name)

    print("Per-group losses:")
    print(results_df)
    print(f"Overall Test Loss: {test_loss}")