import optuna
import pandas as pd
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from utils import create_stratified_splits

# Load main dataset
df = pd.read_csv("./data.csv")

# Create splits
train_df, dev_df, test_df = create_stratified_splits(
    df,
    measurement_col='measurement_type',
    enzyme_col='enzyme_name',
    test_per_class=60,
    dev_per_class=60,
    pki_only=False,
    pki_only_dev_test=True
)

# To maintain correct test split, we need to add in the features post-hoc
# Load the extra rdkit features
feature_df = pd.read_csv('./enzyme_dataset_features.csv')

# Merge on index after splits to keep the same test set
train_df = pd.merge(train_df, feature_df, left_index=True, right_index=True, how='left')
dev_df = pd.merge(dev_df, feature_df, left_index=True, right_index=True, how='left')
test_df = pd.merge(test_df, feature_df, left_index=True, right_index=True, how='left')

# One-hot encode measurement_type and enzyme_name
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder = encoder.fit(df[['measurement_type', 'enzyme_name']])
train_encoded = encoder.transform(train_df[['measurement_type', 'enzyme_name']])
dev_encoded = encoder.transform(dev_df[['measurement_type', 'enzyme_name']])
test_encoded = encoder.transform(test_df[['measurement_type', 'enzyme_name']])

# Get the column names for the encoded features
encoded_columns = encoder.get_feature_names_out(['measurement_type', 'enzyme_name'])

# Append new columns to the data
train_features = pd.DataFrame(train_encoded, columns=encoded_columns)
dev_features = pd.DataFrame(dev_encoded, columns=encoded_columns)
test_features = pd.DataFrame(test_encoded, columns=encoded_columns)

# Prepare targets
train_y = train_df['measurement_value']
dev_y = dev_df['measurement_value']
test_y = test_df['measurement_value']

def objective(trial):
    # Set hyperparameter search ranges for GradientBoostingRegressor
    n_estimators = trial.suggest_int('n_estimators', 20, 200)
    max_depth = trial.suggest_int('max_depth', 3, 30)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)

        model.fit(train_features, train_y)
        preds_dev = model.predict(dev_features)
        dev_mse = mean_squared_error(dev_y, preds_dev)
        mlflow.log_metric("dev_mse", dev_mse)
        mlflow.sklearn.log_model(model, "gradient_boosting_model")

    return dev_mse

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("GB architecture search")

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")