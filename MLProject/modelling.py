import json
import os
from pathlib import Path

import dagshub
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
MLFLOW_URI = "http://127.0.0.1:5000"
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
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def get_dagshub_token() -> str:
    token = os.getenv("DAGSHUB_USER_TOKEN") or os.getenv("DAGSHUB_TOKEN")
    if token:
        return token

    token = input("Masukkan DagsHub Token Anda untuk menyimpan artefak online: ").strip()
    if not token:
        raise ValueError("Token DagsHub tidak boleh kosong.")
    return token

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "Id" in df.columns:
        df = df.drop("Id", axis=1)

    cat_cols_with_na = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    ]
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

    df = pd.get_dummies(df)
    return df


def prepare_data():
    from kaggle.api.kaggle_api_extended import KaggleApi
    import zipfile

    competition = "house-prices-advanced-regression-techniques"
    clean_csv = BASE_DIR / "house-price-dataset_preprocessing.csv"
    if clean_csv.exists():
        df = pd.read_csv(clean_csv)
        if df.select_dtypes(include=[object]).empty:
            return
        print("Dataset sudah ada tetapi mengandung fitur kategorikal. Memproses ulang data...")
    else:
        print("Mengunduh dan memproses data...")
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(competition, path=BASE_DIR)

        with zipfile.ZipFile(BASE_DIR / f"{competition}.zip", "r") as zip_ref:
            zip_ref.extractall(BASE_DIR)

        df = pd.read_csv(BASE_DIR / "train.csv")

    df = preprocess_dataframe(df)
    df.to_csv(clean_csv, index=False)

    raw_files = [
        BASE_DIR / "train.csv",
        BASE_DIR / "test.csv",
        BASE_DIR / "sample_submission.csv",
        BASE_DIR / f"{competition}.zip",
    ]
    for raw_file in raw_files:
        if raw_file.exists():
            os.remove(raw_file)

def configure_tracking() -> str:
    local_uri = MLFLOW_URI
    remote_uri = f"https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow"

    load_env_file(BASE_DIR / ".env")
    token = get_dagshub_token()
    os.environ["DAGSHUB_USER_TOKEN"] = token
    os.environ["DAGSHUB_TOKENwith mlflow.start_run"] = token

    try:
        dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True, dvc=False, patch_mlflow=True)
        mlflow.set_tracking_uri(remote_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        print(f"MLflow Tracking diset ke DagsHub: {remote_uri}")
        return remote_uri
    except Exception as exc:
        print(f"Tidak bisa terhubung ke DagsHub: {exc}")
        print(f"Menggunakan MLflow lokal: {local_uri}")
        mlflow.set_tracking_uri(local_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return local_uri


def save_feature_importance(model: RandomForestRegressor, feature_names: list[str], output_path: Path) -> tuple[Path, Path]:
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    feature_importance_df = feature_importance_df.sort_values("importance", ascending=False)
    csv_path = BASE_DIR / "house-price-dataset_preprocessing.csv"
    feature_importance_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df["feature"].head(15).tolist(), feature_importance_df["importance"].head(15).tolist())
    plt.title("Top 15 Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return csv_path, output_path


def save_predictions(predictions: pd.Series, y_test: pd.Series, output_path: Path) -> None:
    evaluation_df = pd.DataFrame({"actual": y_test, "predicted": predictions})
    evaluation_df.to_csv(output_path, index=False)


def save_metadata(metadata: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


load_env_file(BASE_DIR / ".env")
tracking_uri = configure_tracking()

csv_filename = "house-price-dataset_preprocessing.csv"
csv_path = BASE_DIR / csv_filename
if not csv_path.exists():
    print(f"Dataset bersih tidak ditemukan di {csv_path}. Memulai proses pengunduhan dan pembersihan data...")
    prepare_data()
else:
    df_tmp = pd.read_csv(csv_path)
    if not df_tmp.select_dtypes(include=[object]).empty:
        print("Dataset sudah ada tetapi berisi fitur kategorikal. Memproses ulang data...")
        df_tmp = preprocess_dataframe(df_tmp)
        df_tmp.to_csv(csv_path, index=False)
    df = df_tmp

if 'df' not in locals():
    df = pd.read_csv(csv_path)

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="baseline_random_forest") as run:
    n_estimators = 100
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    artifact_dir = BASE_DIR / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    importance_path = artifact_dir / "feature_importance.png"
    predictions_path = artifact_dir / "predictions.csv"
    config_path = artifact_dir / "model_config.json"

    feature_csv_path, feature_png_path = save_feature_importance(model, X.columns.tolist(), importance_path)
    save_predictions(pd.Series(predictions), y_test, predictions_path)

    model_config = {"model_type": "RandomForestRegressor", "n_estimators": n_estimators, "random_state": 42, "test_size": 0.2}
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(model_config, file, indent=2)

    metadata_path = artifact_dir / "training_metadata.json"
    save_metadata(
        {
            "tracking_uri": tracking_uri,
            "experiment_name": EXPERIMENT_NAME,
            "run_name": "baseline_random_forest",
            "metrics": {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2},
            "artifact_count": 5,
        },
        metadata_path,
    )

    mlflow.log_artifact(str(feature_csv_path))
    mlflow.log_artifact(str(feature_png_path))
    mlflow.log_artifact(str(predictions_path))
    mlflow.log_artifact(str(config_path))
    mlflow.log_artifact(str(metadata_path))
    mlflow.sklearn.log_model(model, "model_house_price")

    print(f"Model berhasil di-training. R2 Score: {r2}")
    print(f"Run ID: {run.info.run_id}")
    print(f"Tracking URI: {tracking_uri}")
    print("Artefak tambahan berhasil disimpan (feature importance, predictions, dan config).")