import os
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
EXPERIMENT_NAME = "House_Price_Prediction"

def configure_tracking():
    dagshub_username = os.getenv("DAGSHUB_USERNAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    
    if dagshub_username and dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        remote_uri = f"https://dagshub.com/{dagshub_username}/Eksperimen_MSML_Yana_Suryana.mlflow"
        mlflow.set_tracking_uri(remote_uri)
    else:
        mlflow.set_tracking_uri(f"file://{BASE_DIR / 'mlruns'}")
    
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
    except mlflow.exceptions.MlflowException:
        mlflow.create_experiment(EXPERIMENT_NAME)
        mlflow.set_experiment(EXPERIMENT_NAME)

def preprocess_dataframe(df):
    if "Id" in df.columns:
        df = df.drop("Id", axis=1)
    
    cat_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
            
    num_cols = {'LotFrontage': df['LotFrontage'].median(), 'MasVnrArea': 0, 'GarageYrBlt': 0}
    for col, val in num_cols.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
            
    if 'SalePrice' in df.columns:
        df['SalePrice'] = np.log1p(df['SalePrice'])
        
    return pd.get_dummies(df)

def prepare_data():
    clean_csv = BASE_DIR / "house-price-dataset_preprocessing.csv"
    if not clean_csv.exists():
        api = KaggleApi()
        api.authenticate()
        competition = "house-prices-advanced-regression-techniques"
        api.competition_download_files(competition, path=BASE_DIR)
        
        with zipfile.ZipFile(BASE_DIR / f"{competition}.zip", "r") as zip_ref:
            zip_ref.extractall(BASE_DIR)
            
        df = preprocess_dataframe(pd.read_csv(BASE_DIR / "train.csv"))
        df.to_csv(clean_csv, index=False)
        
        for f in [BASE_DIR / "train.csv", BASE_DIR / "test.csv", BASE_DIR / "sample_submission.csv", BASE_DIR / f"{competition}.zip"]:
            if f.exists(): os.remove(f)

def run_training():
    configure_tracking()
    prepare_data()
    ARTIFACT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(BASE_DIR / "house-price-dataset_preprocessing.csv")
    X, y = df.drop("SalePrice", axis=1), df["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    mlflow.log_params({"model_type": "RandomForestRegressor", "n_estimators": 100})
    mlflow.log_metrics({"mse": mean_squared_error(y_test, preds), "r2": r2_score(y_test, preds)})
    
    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "local"
    pred_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})
    pred_df.to_csv(ARTIFACT_DIR / f"predictions_{run_id}.csv", index=False)
    
    mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    run_training()
