import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model_uri = 'runs:/8ac20f5615da41f0852b29fcab2784c1/gradient_boosting_model'

model = mlflow.pyfunc.load_model(model_uri)

sklearn_model = model._model_impl.sklearn_model # Access the underlying sklearn model

n_features = sklearn_model.n_features_in_
print(f"n_features_in_ (from pyfunc): {n_features}")