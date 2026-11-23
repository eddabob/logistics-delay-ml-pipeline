import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_and_log(pdf:pd.DataFrame, experiment_name:str = "/Shared/logistics-delay-ml"):
    mlflow.set_experiment(experiment_name)

    categorical_cols = ["carrier", "weather"]
    numeric_cols = ["distance_km", "weight_kg", "expected_hours",
        "dispatch_hour", "is_weekend", "is_cross_region", "weight_per_km", "is_severe_weather"]
    feature_cols = categorical_cols + numeric_cols
    target_col = "is_delayed"

    X = pdf[feature_cols]
    y = pdf[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers = [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    params = {"n_estimators":100, "max_depth":8, "random_state":42}

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifer", RandomForestClassifier(**params))
        ]
    )

    with mlflow.start_run(run_name="rf-baseline"):
        mlflow.log_params(params)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:,1]

        metrics = {
            "accuracy":accuracy_score(y_test, preds),
            "f1_score":f1_score(y_test, preds),
            "roc_auc":roc_auc_score(y_test, probs)
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model = pipeline, name = "delay_classifier",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
            )

        print(f"Run logged successfully. Metrics:{metrics}")

        return pipeline, metrics