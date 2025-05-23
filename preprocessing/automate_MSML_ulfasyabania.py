import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def preprocess_boston_housing(input_csv: str, target_column: str = 'MEDV') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Membaca file CSV Boston Housing, melakukan imputasi missing value dan standarisasi fitur numerik.
    Mengembalikan fitur (X) dan target (y) yang siap dilatih.
    """
    # Baca data
    df = pd.read_csv(input_csv)

    # Imputasi missing value (mean untuk numerik)
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Pisahkan fitur dan target
    X = df_imputed.drop(columns=[target_column])
    y = df_imputed[target_column]

    # Standarisasi fitur numerik
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y

# Contoh penggunaan:
# X, y = preprocess_boston_housing('preprocessing/BostonHousing_preprocessed.csv')
# print(X.head())
# print(y.head())
