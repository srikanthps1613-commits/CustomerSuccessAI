
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

FEATURES = [
    "industry",
    "plan",
    "contract_type",
    "monthly_recurring_revenue",
    "tenure_months",
    "logins_last_30d",
    "usage_change_pct",
    "active_features",
    "total_features",
    "support_tickets_30d",
    "unresolved_tickets",
    "avg_csat",
    "nps",
    "days_since_csm_contact",
    "days_to_renewal"
]

TARGET = "churned"


# --------------------------------------------------
# MODEL TRAINING AND EVALUATION
# --------------------------------------------------

def train_churn_model(df):

    X = df[FEATURES]
    y = df[TARGET]

    # Split 1: 80% development, 20% final testing
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Split 2: 60% training, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=42,
        stratify=y_dev
    )

    # --------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------

    categorical_features = [
        "industry",
        "plan",
        "contract_type"
    ]

    numeric_features = [
        column for column in FEATURES
        if column not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numeric",
                "passthrough",
                numeric_features
            )
        ]
    )

    # --------------------------------------------------
    # RANDOM FOREST PIPELINE
    # --------------------------------------------------

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    # --------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------

    model.fit(X_train, y_train)

    # --------------------------------------------------
    # VALIDATION AND THRESHOLD OPTIMIZATION
    # --------------------------------------------------

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    precision_curve, recall_curve, thresholds = (
        precision_recall_curve(
            y_val,
            val_probabilities
        )
    )

    beta = 2

    f2_scores = (
        (1 + beta ** 2)
        * precision_curve[:-1]
        * recall_curve[:-1]
        /
        (
            beta ** 2 * precision_curve[:-1]
            + recall_curve[:-1]
            + 1e-10
        )
    )

    best_index = int(np.argmax(f2_scores))

    best_threshold = float(
        thresholds[best_index]
    )

    # --------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    confusion = confusion_matrix(
        y_test,
        test_predictions
    )

    # --------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------

    metrics = {

        # Dataset sizes
        "train_size": int(len(X_train)),
        "validation_size": int(len(X_val)),
        "test_size": int(len(X_test)),

        # Threshold
        "decision_threshold": float(best_threshold),

        # Validation performance
        "validation_f2": float(
            f2_scores[best_index]
        ),

        # Test performance
        "accuracy": float(
            accuracy_score(
                y_test,
                test_predictions
            )
        ),

        "precision": float(
            precision_score(
                y_test,
                test_predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                test_predictions,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_test,
                test_predictions,
                zero_division=0
            )
        ),

        "f2_score": float(
            fbeta_score(
                y_test,
                test_predictions,
                beta=2,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_test,
                test_probabilities
            )
        ),

        "average_precision": float(
            average_precision_score(
                y_test,
                test_probabilities
            )
        ),

        "confusion_matrix": confusion.tolist(),

        "classification_report": classification_report(
            y_test,
            test_predictions,
            zero_division=0
        )
    }

    return model, metrics, best_threshold


# --------------------------------------------------
# CHURN PREDICTION
# --------------------------------------------------

def predict_churn(model, df):

    probabilities = model.predict_proba(
        df[FEATURES]
    )[:, 1]

    return probabilities
