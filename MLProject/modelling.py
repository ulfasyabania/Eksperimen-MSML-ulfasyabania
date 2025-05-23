import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

# Pastikan path preprocessing benar untuk runner Linux/GitHub Actions
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
if dir_path not in sys.path:
    sys.path.append(dir_path)
from automate_MSML_ulfasyabania import preprocess_boston_housing

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default='bostonhousing_preprocessing.csv')
    args = parser.parse_args()

    try:
        print(f"[INFO] Loading data from: {args.input_csv}")
        X, y = preprocess_boston_housing(args.input_csv, target_column='MEDV')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        mlflow.set_experiment("BostonHousing_RF")
        with mlflow.start_run() as run:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(model, "model")
            test_df = pd.concat([X_test, y_test], axis=1)
            test_df.to_csv("test_data.csv", index=False)
            mlflow.log_artifact("test_data.csv")
            print(f"Run ID: {run.info.run_id}")
            print(f"MSE: {mse}")
            print(f"R2: {r2}")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Docker Hub link: https://hub.docker.com/r/ulfasyabania173/mlflow-bostonhousing
