import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_rf_model_and_return(data_path, feature_cols):
    """Rebuild df_feat and train RF model once."""
    df = pd.read_csv(data_path)
    df = df.sort_values(["user_id", "day"]).reset_index(drop=True)

    df_feat = df.copy()

    # Rolling features
    df_feat["avg_steps_7d"] = (
        df_feat.groupby("user_id")["steps"]
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df_feat["avg_sleep_7d"] = (
        df_feat.groupby("user_id")["sleep_hours"]
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df_feat["avg_stress_7d"] = (
        df_feat.groupby("user_id")["stress_level"]
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Flags (mirroring notebook)
    df_feat["low_sleep_flag"] = (df_feat["sleep_hours"] < 6).astype(int)
    df_feat["high_stress_flag"] = (df_feat["stress_level"] >= 7).astype(int)
    df_feat["low_steps_flag"] = (df_feat["steps"] < 6000).astype(int)
    df_feat["low_water_flag"] = (df_feat["water_glasses"] < 6).astype(int)
    df_feat["high_fatigue_flag"] = (df_feat["fatigue_score"] >= 7).astype(int)

    ml_df = df_feat.dropna(subset=feature_cols).copy()
    X = ml_df[feature_cols]
    y = ml_df["high_fatigue_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    rf.fit(X_train, y_train)

    return rf
