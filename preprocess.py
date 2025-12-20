import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Final feature order expected by the trained model
FEATURE_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region_northwest",
    "region_southeast",
    "region_southwest"
]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw insurance data into model-ready format.
    This function is used BOTH during training and prediction.
    """

    df = df.copy()

    # Label Encoding for binary categorical columns
    df["sex"] = LabelEncoder().fit_transform(df["sex"])
    df["smoker"] = LabelEncoder().fit_transform(df["smoker"])

    # One-hot encoding for region
    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    # Ensure all required columns exist (important for single-row prediction)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[FEATURE_COLUMNS]

    return df
