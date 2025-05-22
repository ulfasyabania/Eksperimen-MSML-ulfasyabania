import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocessing')))
from MLProject import automate_MSML_ulfasyabania.py


def train_and_log_model(input_csv: str, target_column: str = 'MEDV'):
    try:
        mlflow.sklearn.autolog()
        # Preprocessing
        X, y = preprocess_boston_housing(input_csv, target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # MLflow experiment setup
        mlflow.set_experiment("BostonHousing_RF")
        with mlflow.start_run() as run:
            # Model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict & metrics
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Logging manual (opsional, autolog sudah otomatis)
            # mlflow.log_param("n_estimators", 100)
            # mlflow.log_param("random_state", 42)
            # mlflow.log_metric("mse", mse)
            # mlflow.log_metric("r2", r2)
            # mlflow.sklearn.log_model(model, "model")
            
            # Logging artifacts (optional: save test set)
            test_df = pd.concat([X_test, y_test], axis=1)
            test_df.to_csv("test_data.csv", index=False)
            mlflow.log_artifact("test_data.csv")
            
            print(f"Run ID: {run.info.run_id}")
            print(f"MSE: {mse}")
            print(f"R2: {r2}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        mlflow.set_experiment("BostonHousing_RF")
        with mlflow.start_run() as run:
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
            with open("mlflow_error.log", "w") as f:
                traceback.print_exc(file=f)
            mlflow.log_artifact("mlflow_error.log")

if __name__ == "__main__":
    train_and_log_model(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BostonHousing_raw.csv')))
