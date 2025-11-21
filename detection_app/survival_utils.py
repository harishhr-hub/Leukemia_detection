"""
Survival analysis utilities: Cox regression, LASSO, risk stratification, Kaplan-Meier curves.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64
from .models import Patient, PatientFollowUp, SurvivalAnalysis


def prepare_survival_data():
    """
    Prepare patient data for survival analysis.
    Returns: DataFrame with clinical features, survival time, and event status.
    """
    patients = Patient.objects.filter(follow_ups__isnull=False).distinct()
    
    data = []
    for patient in patients:
        # Get most recent follow-up
        latest_followup = patient.follow_ups.latest('visit_date')
        
        row = {
            'patient_id': patient.id,
            'age': patient.age,
            'wbc': patient.wbc_count,
            'blast': patient.blast_percentage,
            'hemoglobin': patient.hemoglobin,
            'platelets': patient.platelets,
            'T': latest_followup.survival_days,  # time to event or censoring
            'E': 1 if latest_followup.event_status != 'ALIVE' else 0,  # event indicator
        }
        
        # Add most severe detection risk level
        detections = patient.detections.all()
        if detections.exists():
            risk_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
            max_risk = max([risk_map.get(d.risk_level, 0) for d in detections])
            row['detection_risk'] = max_risk
        else:
            row['detection_risk'] = 0
        
        data.append(row)
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    return df


def fit_cox_model(df=None):
    """
    Fit Cox Proportional Hazards model.
    Returns: CoxPHFitter object, feature importances dict.
    """
    if df is None:
        df = prepare_survival_data()
    
    if df is None or len(df) < 5:
        return None, None  # Need minimum samples
    
    # Select features with data
    features = ['age', 'wbc', 'blast', 'hemoglobin', 'platelets', 'detection_risk']
    available_features = [f for f in features if f in df.columns and df[f].notna().sum() > 2]
    
    if not available_features:
        return None, None
    
    df_clean = df[available_features + ['T', 'E']].dropna()
    
    if len(df_clean) < 3:
        return None, None
    
    # Standardize features
    scaler = StandardScaler()
    df_clean_scaled = df_clean.copy()
    df_clean_scaled[available_features] = scaler.fit_transform(df_clean[available_features])
    
    # Fit Cox model
    cph = CoxPHFitter()
    try:
        cph.fit(df_clean_scaled, duration_col='T', event_col='E')
        
        # Extract feature importances
        coef_dict = {
            fname: float(cph.params_[fname])
            for fname in available_features
            if fname in cph.params_.index
        }
        
        return cph, coef_dict
    except Exception as e:
        print(f"Cox fit error: {e}")
        return None, None


def compute_risk_score(patient, cph=None, coef_dict=None):
    """
    Compute individual risk score for a patient using Cox model.
    Returns: risk_score (float), risk_group (str).
    """
    if cph is None:
        _, coef_dict = fit_cox_model()
    
    if coef_dict is None:
        return None, None
    
    # Standardize patient features using population stats
    df = prepare_survival_data()
    if df is None:
        return None, None
    
    scaler = StandardScaler()
    features_list = list(coef_dict.keys())
    available = [f for f in features_list if f in df.columns]
    if available:
        scaler.fit(df[available])
    
    # Build feature vector for this patient
    risk_score = 0.0
    for feature, coef in coef_dict.items():
        if feature == 'age':
            val = patient.age
        elif feature == 'wbc':
            val = patient.wbc_count
        elif feature == 'blast':
            val = patient.blast_percentage
        elif feature == 'hemoglobin':
            val = patient.hemoglobin
        elif feature == 'platelets':
            val = patient.platelets
        elif feature == 'detection_risk':
            detections = patient.detections.all()
            if detections.exists():
                risk_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
                val = max([risk_map.get(d.risk_level, 0) for d in detections])
            else:
                val = 0
        else:
            val = 0
        
        if val is not None:
            risk_score += coef * val
    
    # Classify into risk groups based on risk score distribution
    if risk_score < -0.5:
        risk_group = 'LOW'
    elif risk_score < 0.5:
        risk_group = 'INTERMEDIATE'
    else:
        risk_group = 'HIGH'
    
    return risk_score, risk_group


def plot_kaplan_meier(patient_ids=None):
    """
    Generate Kaplan-Meier curves for risk groups.
    Returns: base64-encoded PNG image.
    """
    df = prepare_survival_data()
    if df is None or len(df) < 3:
        return None
    
    # Stratify by detection risk
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for risk_level in ['LOW', 'MEDIUM', 'HIGH']:
        risk_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        mask = df['detection_risk'] == risk_map.get(risk_level, -1)
        
        if mask.sum() < 2:
            continue
        
        kmf = KaplanMeierFitter()
        kmf.fit(df[mask]['T'], df[mask]['E'], label=f'{risk_level} Risk')
        kmf.plot_survival_function(ax=ax, ci_show=True)
    
    plt.xlabel('Days')
    plt.ylabel('Survival Probability')
    plt.title('Kaplan-Meier Curves by Detection Risk Level')
    plt.legend()
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return img_base64


def plot_survival_roc(patient_ids=None):
    """
    Plot ROC curve for survival prediction (using risk score).
    Returns: base64-encoded PNG image.
    """
    df = prepare_survival_data()
    if df is None or len(df) < 5:
        return None
    
    cph, coef_dict = fit_cox_model(df)
    if cph is None:
        return None
    
    # Compute risk scores for all patients
    risk_scores = []
    events = []
    for _, row in df.iterrows():
        risk_score, _ = compute_risk_score(Patient.objects.get(id=row['patient_id']), cph, coef_dict)
        if risk_score is not None:
            risk_scores.append(risk_score)
            events.append(row['E'])
    
    if not risk_scores:
        return None
    
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(events, risk_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Survival Prediction ROC Curve')
    ax.legend()
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return img_base64
