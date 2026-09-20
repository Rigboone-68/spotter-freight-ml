from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
]

NUMERIC_FEATURES = [
    "distance",
    "weight",
    "day_of_week",
    "day_of_year",
    "day_of_year_sin",
    "day_of_year_cos",
    "days_since_start",
]

GEOGRAPHIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]

MODEL_FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
    + GEOGRAPHIC_FEATURES
)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the final model features without using target information.
    """
    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])

    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear

    data["day_of_year_sin"] = np.sin(
        2 * np.pi * data["day_of_year"] / 365.25
    )

    data["day_of_year_cos"] = np.cos(
        2 * np.pi * data["day_of_year"] / 365.25
    )

    data["days_since_start"] = (
        data["date"] - pd.Timestamp("2025-01-01")
    ).dt.days

    return data


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_final_model() -> Pipeline:
    """
    Build the frozen Candidate 5C model.
    """

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_transformer,
            NUMERIC_FEATURES + GEOGRAPHIC_FEATURES,
        ),
        (
            "categorical",
            categorical_transformer,
            CATEGORICAL_FEATURES,
        ),
    ])

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=5.0,
        random_state=42,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# December geographic reconstruction
# ---------------------------------------------------------------------------

def add_december_coordinates(
    december: pd.DataFrame,
    development: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct December pickup and delivery coordinates from
    verified development-data city mappings.
    """

    pickup_coordinates = (
        development[
            ["pickup", "pickup_lat", "pickup_lon"]
        ]
        .drop_duplicates("pickup")
        .set_index("pickup")
    )

    delivery_coordinates = (
        development[
            ["delivery", "delivery_lat", "delivery_lon"]
        ]
        .drop_duplicates("delivery")
        .set_index("delivery")
    )

    data = december.copy()

    data["pickup_lat"] = data["pickup"].map(
        pickup_coordinates["pickup_lat"]
    )

    data["pickup_lon"] = data["pickup"].map(
        pickup_coordinates["pickup_lon"]
    )

    data["delivery_lat"] = data["delivery"].map(
        delivery_coordinates["delivery_lat"]
    )

    data["delivery_lon"] = data["delivery"].map(
        delivery_coordinates["delivery_lon"]
    )

    return data


# ---------------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------------

def train_final_model(
    development: pd.DataFrame,
) -> Pipeline:
    """
    Train the frozen final model on the complete development dataset.
    """

    features = create_features(development)

    X = features[MODEL_FEATURES]
    y = development["posted_rate"]

    model = build_final_model()
    model.fit(X, y)

    return model


def predict_validation(
    model: Pipeline,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate predictions for every validation row.
    """

    features = create_features(validation)

    predictions = model.predict(
        features[MODEL_FEATURES]
    )

    return pd.DataFrame({
        "load_id": validation["load_id"],
        "predicted_rate": predictions,
    })


def predict_december(
    model: Pipeline,
    december: pd.DataFrame,
    development: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate predictions for the fixed December inputs.
    """

    data = add_december_coordinates(
        december,
        development,
    )

    features = create_features(data)

    predictions = model.predict(
        features[MODEL_FEATURES]
    )

    output = data[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
        ]
    ].copy()

    output["predicted_rate"] = predictions

    return output