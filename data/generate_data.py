"""
Mommy Care - Synthetic Maternal Health Dataset Generator
Generates training data based on clinical patterns from UCI Maternal Health Risk Dataset
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def generate_maternal_health_data(n_samples=2000, random_state=42):
    np.random.seed(random_state)

    # --- Low Risk Group (~50%) ---
    n_low = int(n_samples * 0.50)
    low = pd.DataFrame({
        'age': np.random.normal(26, 4, n_low).clip(18, 35).astype(int),
        'systolic_bp': np.random.normal(112, 8, n_low).clip(90, 129),
        'diastolic_bp': np.random.normal(74, 6, n_low).clip(60, 84),
        'blood_glucose': np.random.normal(88, 8, n_low).clip(70, 105),
        'body_temp': np.random.normal(36.6, 0.2, n_low).clip(36.0, 37.2),
        'heart_rate': np.random.normal(76, 7, n_low).clip(60, 95),
        'weight_gain_kg': np.random.normal(9, 3, n_low).clip(4, 16),
        'gestational_age_weeks': np.random.randint(8, 40, n_low),
        'previous_pregnancies': np.random.choice([0, 1, 2], n_low, p=[0.4, 0.4, 0.2]),
        'previous_complications': np.zeros(n_low, dtype=int),
        'risk_level': ['low'] * n_low
    })

    # --- Mid Risk Group (~30%) ---
    n_mid = int(n_samples * 0.30)
    mid = pd.DataFrame({
        'age': np.random.normal(32, 5, n_mid).clip(25, 42).astype(int),
        'systolic_bp': np.random.normal(133, 7, n_mid).clip(120, 149),
        'diastolic_bp': np.random.normal(86, 6, n_mid).clip(80, 99),
        'blood_glucose': np.random.normal(118, 15, n_mid).clip(100, 155),
        'body_temp': np.random.normal(37.0, 0.3, n_mid).clip(36.5, 38.0),
        'heart_rate': np.random.normal(84, 8, n_mid).clip(70, 105),
        'weight_gain_kg': np.random.normal(14, 3, n_mid).clip(8, 20),
        'gestational_age_weeks': np.random.randint(16, 40, n_mid),
        'previous_pregnancies': np.random.choice([0, 1, 2, 3], n_mid, p=[0.2, 0.3, 0.3, 0.2]),
        'previous_complications': np.random.choice([0, 1], n_mid, p=[0.6, 0.4]),
        'risk_level': ['mid'] * n_mid
    })

    # --- High Risk Group (~20%) ---
    n_high = n_samples - n_low - n_mid
    high = pd.DataFrame({
        'age': np.random.normal(38, 4, n_high).clip(35, 48).astype(int),
        'systolic_bp': np.random.normal(152, 10, n_high).clip(140, 180),
        'diastolic_bp': np.random.normal(96, 8, n_high).clip(88, 120),
        'blood_glucose': np.random.normal(160, 20, n_high).clip(140, 200),
        'body_temp': np.random.normal(37.5, 0.4, n_high).clip(37.0, 39.5),
        'heart_rate': np.random.normal(94, 10, n_high).clip(80, 120),
        'weight_gain_kg': np.random.normal(18, 2, n_high).clip(12, 22),
        'gestational_age_weeks': np.random.randint(20, 40, n_high),
        'previous_pregnancies': np.random.choice([0, 1, 2, 3], n_high, p=[0.1, 0.2, 0.3, 0.4]),
        'previous_complications': np.random.choice([0, 1], n_high, p=[0.2, 0.8]),
        'risk_level': ['high'] * n_high
    })

    df = pd.concat([low, mid, high], ignore_index=True)
    df = shuffle(df, random_state=random_state).reset_index(drop=True)

    # Round floats
    for col in ['systolic_bp', 'diastolic_bp', 'blood_glucose', 'body_temp', 'heart_rate', 'weight_gain_kg']:
        df[col] = df[col].round(1)

    return df


if __name__ == "__main__":
    df = generate_maternal_health_data(2000)
    df.to_csv("maternal_health_data.csv", index=False)
    print(df['risk_level'].value_counts())
    print(df.describe())
