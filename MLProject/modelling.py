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
    remote_uri = f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('DAGSHUB_REPO')}.mlflow"
    if os.getenv("DAGSHUB_USERNAME") and os.getenv("DAGSHUB_TOKEN"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
        mlflow.set_tracking_uri(remote_uri)
    else:
        mlflow.set_tracking_uri(f"file://{BASE_DIR / 'mlruns'}")
    mlflow.set_experiment(EXPERIMENT_NAME)

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "Id" in df.columns:
        df = df.drop("Id", axis=1)
    cat_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
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

configure_tracking()
prepare_data()
ARTIFACT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(BASE_DIR / "house-price-dataset_preprocessing.csv")
X, y = df.drop("SalePrice", axis=1), df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="baseline_random_forest") as run:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mlflow.log_params({"model_type": "RandomForestRegressor", "n_estimators": 100})
    mlflow.log_metrics({"mse": mean_squared_error(y_test, preds), "r2": r2_score(y_test, preds)})
    
    pred_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})
    pred_df.to_csv(ARTIFACT_DIR / f"predictions_{run.info.run_id}.csv", index=False)
    
    mlflow.sklearn.log_model(model, "model")
