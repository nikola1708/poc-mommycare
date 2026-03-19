"""
Mommy Care - ML Model Training & Prediction Engine
Models: Random Forest, Logistic Regression, Gradient Boosting
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    'age', 'systolic_bp', 'diastolic_bp', 'blood_glucose',
    'body_temp', 'heart_rate', 'weight_gain_kg',
    'gestational_age_weeks', 'previous_pregnancies', 'previous_complications'
]
TARGET_COL = 'risk_level'
LABEL_ORDER = ['low', 'mid', 'high']


def load_or_generate_data():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "maternal_health_data.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    # Generate on the fly
    from data.generate_data import generate_maternal_health_data
    df = generate_maternal_health_data(2000)
    df.to_csv(data_path, index=False)
    return df


def train_all_models():
    """Train all three models and save to disk."""
    df = load_or_generate_data()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    le = LabelEncoder()
    le.classes_ = np.array(LABEL_ORDER)
    y_enc = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=8,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, solver='lbfgs'
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08,
            max_depth=4, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        use_scaled = name in ("Logistic Regression",)
        Xtr = X_train_sc if use_scaled else X_train
        Xte = X_test_sc if use_scaled else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(
            model,
            X_train_sc if use_scaled else X_train,
            y_train, cv=5, scoring='accuracy'
        )

        results[name] = {
            "model": model,
            "accuracy": acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=LABEL_ORDER, output_dict=True
            )
        }

    # Persist
    bundle = {
        "scaler": scaler,
        "label_encoder": le,
        "models": {k: v["model"] for k, v in results.items()},
        "metrics": {k: {ky: v for ky, v in vv.items() if ky != "model"} for k, vv in results.items()},
        "feature_cols": FEATURE_COLS,
        "label_order": LABEL_ORDER,
    }
    with open(os.path.join(MODELS_DIR, "bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    return bundle


def load_bundle():
    path = os.path.join(MODELS_DIR, "bundle.pkl")
    if not os.path.exists(path):
        return train_all_models()
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_risk(input_dict: dict, model_name: str = "Random Forest"):
    """
    Returns: {
        'risk_level': 'low'|'mid'|'high',
        'probabilities': {'low': float, 'mid': float, 'high': float},
        'preeclampsia_risk': float,
        'gd_risk': float,
        'cs_risk': float,
    }
    """
    bundle = load_bundle()
    model = bundle["models"][model_name]
    scaler = bundle["scaler"]
    le = bundle["label_encoder"]

    row = pd.DataFrame([{col: input_dict.get(col, 0) for col in FEATURE_COLS}])

    use_scaled = model_name in ("Logistic Regression",)
    X = scaler.transform(row) if use_scaled else row

    pred_enc = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    risk_label = le.inverse_transform([pred_enc])[0]
    prob_map = {LABEL_ORDER[i]: round(float(proba[i]), 4) for i in range(len(LABEL_ORDER))}

    # Clinical sub-risk estimates (rule-enhanced)
    sbp = input_dict.get('systolic_bp', 110)
    dbp = input_dict.get('diastolic_bp', 70)
    gluc = input_dict.get('blood_glucose', 90)
    age = input_dict.get('age', 28)
    wt = input_dict.get('weight_gain_kg', 9)

    preec_base = prob_map['high'] * 0.55 + prob_map['mid'] * 0.18
    if sbp >= 140 or dbp >= 90:
        preec_base = min(preec_base + 0.25, 0.95)
    if age >= 35:
        preec_base = min(preec_base + 0.08, 0.95)

    gd_base = prob_map['high'] * 0.45 + prob_map['mid'] * 0.20
    if gluc >= 140:
        gd_base = min(gd_base + 0.30, 0.95)
    elif gluc >= 126:
        gd_base = min(gd_base + 0.15, 0.95)

    cs_base = prob_map['high'] * 0.50 + prob_map['mid'] * 0.22 + 0.12
    if age >= 35:
        cs_base = min(cs_base + 0.10, 0.95)
    if wt > 16:
        cs_base = min(cs_base + 0.08, 0.95)

    return {
        "risk_level": risk_label,
        "probabilities": prob_map,
        "preeclampsia_risk": round(preec_base, 3),
        "gd_risk": round(gd_base, 3),
        "cs_risk": round(cs_base, 3),
    }


def get_feature_importance(model_name: str = "Random Forest"):
    bundle = load_bundle()
    model = bundle["models"][model_name]
    if hasattr(model, "feature_importances_"):
        fi = dict(zip(FEATURE_COLS, model.feature_importances_))
        return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    return {}


if __name__ == "__main__":
    bundle = train_all_models()
    for name, metrics in bundle["metrics"].items():
        print(f"\n=== {name} ===")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  CV Mean  : {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
