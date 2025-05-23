import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocessing')))
from automate_MSML_ulfasyabania import preprocess_boston_housing

def train_and_log_model_with_tuning(input_csv: str, target_column: str = 'MEDV'):
    # Preprocessing
    X, y = preprocess_boston_housing(input_csv, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }

    mlflow.set_tracking_uri("https://dagshub.com/ulfasyabania/Eksperimen_MSML_ulfasyabania.mlflow")
    mlflow.set_experiment("BostonHousing_RF_Tuning")
    with mlflow.start_run() as run:
        # Grid Search
        grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        # Predict & metrics
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        train_score = best_model.score(X_train, y_train)
        test_score = best_model.score(X_test, y_test)

        # Manual logging
        mlflow.log_param("best_params", grid.best_params_)
        mlflow.log_param("cv_results_mean_test_score", grid.cv_results_["mean_test_score"][grid.best_index_])
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("test_score", test_score)
        mlflow.sklearn.log_model(best_model, "model")

        # Logging artifacts (optional: save test set)
        test_df = pd.concat([X_test, y_test], axis=1)
        test_df.to_csv("test_data_tuning.csv", index=False)
        mlflow.log_artifact("test_data_tuning.csv")
        
        print(f"Run ID: {run.info.run_id}")
        print(f"Best Params: {grid.best_params_}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2: {r2}")
        print(f"Train Score: {train_score}")
        print(f"Test Score: {test_score}")

if __name__ == "__main__":
    os.environ["MLFLOW_TRACKING_USERNAME"] = "ulfasyabania"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "952cff7a88b41467aed8c87282ec250bce0e6051"
    train_and_log_model_with_tuning(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BostonHousing_raw.csv')))
