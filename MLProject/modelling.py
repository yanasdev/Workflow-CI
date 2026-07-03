import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = "House_Price_Prediction"
MLFLOW_LOCAL_URI = f"file://{BASE_DIR / 'mlruns'}"
DAGSHUB_OWNER = "yanas.dev"
DAGSHUB_REPO = "Eksperimen_MSML_Yana_Suryana"

def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

def configure_tracking() -> str:
    load_env_file(BASE_DIR / ".env")
    remote_uri = f"https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow"
    mlflow_tracking_username = os.getenv("DAGSHUB_USERNAME") or os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_tracking_password = os.getenv("DAGSHUB_TOKEN") or os.getenv("MLFLOW_TRACKING_PASSWORD")
    if not mlflow_tracking_username or not mlflow_tracking_password:
        mlflow.set_tracking_uri(MLFLOW_LOCAL_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return MLFLOW_LOCAL_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_tracking_password
    mlflow.set_tracking_uri(remote_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return remote_uri

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "Id" in df.columns:
        df = df.drop("Id", axis=1)
    cat_cols_with_na = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in cat_cols_with_na:
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
    from kaggle.api.kaggle_api_extended import KaggleApi
    import zipfile
    competition = "house-prices-advanced-regression-techniques"
    clean_csv = BASE_DIR / "house-price-dataset_preprocessing.csv"
    if not clean_csv.exists():
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(competition, path=BASE_DIR)
        with zipfile.ZipFile(BASE_DIR / f"{competition}.zip", "r") as zip_ref:
            zip_ref.extractall(BASE_DIR)
        df = preprocess_dataframe(pd.read_csv(BASE_DIR / "train.csv"))
        df.to_csv(clean_csv, index=False)
        for f in [BASE_DIR / "train.csv", BASE_DIR / "test.csv", BASE_DIR / "sample_submission.csv", BASE_DIR / f"{competition}.zip"]:
            if f.exists(): os.remove(f)

load_env_file(BASE_DIR / ".env")
tracking_uri = configure_tracking()
prepare_data()
df = pd.read_csv(BASE_DIR / "house-price-dataset_preprocessing.csv")

X, y = df.drop("SalePrice", axis=1), df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if "MLFLOW_RUN_ID" in os.environ:
    del os.environ["MLFLOW_RUN_ID"]
    
if mlflow.active_run(): mlflow.end_run()

with mlflow.start_run(run_name="baseline_random_forest") as run:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mlflow.log_params({"model_type": "RandomForestRegressor", "n_estimators": 100})
    mlflow.log_metrics({"mse": mean_squared_error(y_test, predictions), "r2": r2_score(y_test, predictions)})
    
    artifact_dir = BASE_DIR / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    model_local_dir = BASE_DIR / "model_export"
    
    mlflow.sklearn.save_model(model, str(model_local_dir))
    mlflow.sklearn.log_model(model, "model")
    
    with (BASE_DIR / "last_run_id.txt").open("w") as f:
        f.write(run.info.run_id)