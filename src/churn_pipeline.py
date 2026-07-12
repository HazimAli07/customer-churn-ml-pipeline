"""Data preparation and model-building utilities for telco churn prediction."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def download_data(destination: Path = DEFAULT_DATA_PATH) -> Path:
    """Download the public IBM sample dataset when it is not available locally."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urlretrieve(DATA_URL, destination)
    return destination


def clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy with consistent strings and numeric TotalCharges."""
    data = frame.copy()
    data.columns = data.columns.str.strip()

    missing_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    } - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    object_columns = data.select_dtypes(include="object").columns
    for column in object_columns:
        data[column] = data[column].str.strip()

    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    unexpected_targets = set(data[TARGET_COLUMN].dropna().unique()) - {"Yes", "No"}
    if unexpected_targets:
        raise ValueError(f"Unexpected churn labels: {sorted(unexpected_targets)}")

    return data


def load_data(path: Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load the data and return model features, binary target, and cleaned frame."""
    data_path = download_data(Path(path))
    cleaned = clean_dataframe(pd.read_csv(data_path))

    features = cleaned.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    target = cleaned[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    if target.isna().any():
        raise ValueError("Target encoding produced missing values.")

    return features, target.astype(int), cleaned


def build_pipeline() -> Pipeline:
    """Build a leakage-safe preprocessing and logistic-regression pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def coefficient_table(fitted_pipeline: Pipeline) -> pd.DataFrame:
    """Return model coefficients paired with their transformed feature names."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_.ravel()

    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )
    table["feature"] = (
        table["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )
    table["absolute_coefficient"] = table["coefficient"].abs()
    return table.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)
