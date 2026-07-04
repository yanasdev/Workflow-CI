import os
import zipfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATASET_FILE = BASE_DIR / "house-price-dataset_preprocessing.csv"

EXPERIMENT_NAME = "House_Price_Prediction"


def configure_experiment():
    if os.getenv("MLFLOW_TRACKING_URI"):
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    else:
        local_uri = f"file://{BASE_DIR / 'mlruns'}"
        mlflow.set_tracking_uri(local_uri)
        print(f"Tracking URI: {local_uri}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Experiment: {EXPERIMENT_NAME}")


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "Id" in df.columns:
        df = df.drop(columns="Id")

    categorical_columns = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
    ]

    for column in categorical_columns:
        if column in df.columns:
            df[column] = df[column].fillna("None")

    numeric_fill_values = {
        "LotFrontage": df["LotFrontage"].median(),
        "MasVnrArea": 0,
        "GarageYrBlt": 0,
    }

    for column, value in numeric_fill_values.items():
        if column in df.columns:
            df[column] = df[column].fillna(value)

    if "SalePrice" in df.columns:
        df["SalePrice"] = np.log1p(df["SalePrice"])

    return pd.get_dummies(df)

def prepare_data():
    if DATASET_FILE.exists():
        print("Using cached preprocessed dataset.")
        return

    print("Downloading dataset from Kaggle...")

    api = KaggleApi()
    api.authenticate()

    competition = "house-prices-advanced-regression-techniques"

    api.competition_download_files(
        competition,
        path=BASE_DIR,
    )

    with zipfile.ZipFile(
        BASE_DIR / f"{competition}.zip",
        "r",
    ) as archive:
        archive.extractall(BASE_DIR)

    dataframe = pd.read_csv(BASE_DIR / "train.csv")
    dataframe = preprocess_dataframe(dataframe)

    dataframe.to_csv(DATASET_FILE, index=False)

    print("Dataset preprocessed and saved.")

    temporary_files = [
        BASE_DIR / "train.csv",
        BASE_DIR / "test.csv",
        BASE_DIR / "sample_submission.csv",
        BASE_DIR / f"{competition}.zip",
    ]

    for file in temporary_files:
        if file.exists():
            file.unlink()


def train_model():
    dataframe = pd.read_csv(DATASET_FILE)

    X = dataframe.drop(columns="SalePrice")
    y = dataframe["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("Training RandomForestRegressor...")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return (
        model,
        predictions,
        y_test.reset_index(drop=True),
    )

def log_results(model, predictions, y_test):
    active_run = mlflow.active_run()

    if active_run is None:
        raise RuntimeError("No active MLflow run found.")

    run_id = active_run.info.run_id

    print(f"Logging artifacts for run: {run_id}")

    mlflow.log_params(
        {
            "model_type": "RandomForestRegressor",
            "n_estimators": 100,
            "random_state": 42,
        }
    )

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    mlflow.log_metrics(
        {
            "mse": float(mse),
            "r2": float(r2),
        }
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    prediction_file = ARTIFACT_DIR / f"predictions_{run_id}.csv"

    pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": predictions,
        }
    ).to_csv(
        prediction_file,
        index=False,
    )

    mlflow.log_artifact(str(prediction_file))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
    )

    print(f"Model logged successfully: {run_id}")

def run_training():
    configure_experiment()

    prepare_data()

    model, predictions, y_test = train_model()

    active_run = mlflow.active_run()

    if active_run is not None:
        print(f"Using existing MLflow run: {active_run.info.run_id}")
        log_results(model, predictions, y_test)
    else:
        print("Starting new MLflow run")
        with mlflow.start_run():
            log_results(model, predictions, y_test)


if __name__ == "__main__":
    run_training()
