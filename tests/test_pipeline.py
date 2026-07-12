"""Regression tests for the churn preprocessing and prediction pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.churn_pipeline import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATA_PATH,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_pipeline,
    clean_dataframe,
    load_data,
)


def sample_features() -> pd.DataFrame:
    rows = []
    for index in range(12):
        row = {
            "SeniorCitizen": index % 2,
            "tenure": 2 + index * 4,
            "MonthlyCharges": 35.0 + index * 4.5,
            "TotalCharges": 70.0 + index * 165.0,
            "gender": "Female" if index % 2 else "Male",
            "Partner": "Yes" if index % 3 else "No",
            "Dependents": "No" if index % 2 else "Yes",
            "PhoneService": "Yes",
            "MultipleLines": "Yes" if index % 2 else "No",
            "InternetService": "Fiber optic" if index % 2 else "DSL",
            "OnlineSecurity": "No" if index % 2 else "Yes",
            "OnlineBackup": "Yes" if index % 3 else "No",
            "DeviceProtection": "No" if index % 2 else "Yes",
            "TechSupport": "No" if index % 2 else "Yes",
            "StreamingTV": "Yes" if index % 2 else "No",
            "StreamingMovies": "No" if index % 2 else "Yes",
            "Contract": "Month-to-month" if index % 2 else "One year",
            "PaperlessBilling": "Yes" if index % 2 else "No",
            "PaymentMethod": "Electronic check" if index % 2 else "Mailed check",
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)


class PipelineTests(unittest.TestCase):
    def test_clean_dataframe_converts_blank_total_charges(self) -> None:
        features = sample_features().head(2)
        frame = features.copy()
        frame.insert(0, ID_COLUMN, ["A", "B"])
        frame[TARGET_COLUMN] = ["No", "Yes"]
        frame["TotalCharges"] = [" ", "125.50"]

        cleaned = clean_dataframe(frame)

        self.assertTrue(pd.isna(cleaned.loc[0, "TotalCharges"]))
        self.assertEqual(cleaned.loc[1, "TotalCharges"], 125.50)

    def test_pipeline_handles_an_unseen_category(self) -> None:
        features = sample_features()
        target = pd.Series([0, 1] * 6)
        model = build_pipeline().fit(features, target)

        unseen = features.iloc[[0]].copy()
        unseen.loc[:, "PaymentMethod"] = "New digital wallet"
        prediction = model.predict(unseen)

        self.assertEqual(prediction.shape, (1,))
        self.assertIn(int(prediction[0]), {0, 1})

    def test_ibm_dataset_shape_and_target(self) -> None:
        if not Path(DEFAULT_DATA_PATH).exists():
            self.skipTest("IBM dataset is not available in this checkout.")

        features, target, cleaned = load_data(DEFAULT_DATA_PATH)

        self.assertEqual(len(cleaned), 7_043)
        self.assertEqual(features.shape[1], 19)
        self.assertFalse(target.isna().any())
        self.assertEqual(set(target.unique()), {0, 1})


if __name__ == "__main__":
    unittest.main()
