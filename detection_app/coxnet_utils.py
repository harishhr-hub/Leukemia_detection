"""
CoxNet: LASSO-Cox Regression for Risk Prediction

This module implements LASSO-regularized Cox proportional hazards regression
for predicting patient survival risk based on clinical features.

References:
    - Tibshirani, R. (1997). The lasso method for variable selection in the Cox model.
    - Simon, N., et al. (2011). Regularization paths for Cox's partial likelihood.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter, LogLogisticAFTFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
import warnings
import pickle
import os
from django.conf import settings

warnings.filterwarnings('ignore')


class CoxNetModel:
    """LASSO-Cox regression model for survival analysis."""
    
    def __init__(self, penalizer=0.1, l1_ratio=1.0, model_type='cph'):
        """
        Initialize CoxNet model.
        
        Args:
            penalizer: L1/L2 regularization strength (lambda)
            l1_ratio: Ratio of L1 to L2 penalty (1.0 for LASSO, 0.0 for Ridge)
            model_type: 'cph' (Cox), 'weibull', or 'loglogistic'
        """
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == 'cph':
            self.model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        elif model_type == 'weibull':
            self.model = WeibullAFTFitter(penalizer=penalizer)
        elif model_type == 'loglogistic':
            self.model = LogLogisticAFTFitter(penalizer=penalizer)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.is_fitted = False
        self.feature_names = None
        self.feature_importances = None
        self.concordance_index_ = None
    
    def fit(self, T, E, X, feature_names=None):
        """
        Fit Cox model to survival data.
        
        Args:
            T: Array of survival times (days, months, etc.)
            E: Array of event indicators (1 = event occurred, 0 = censored)
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            
        Returns:
            self
        """
        if X.shape[0] != len(T) or X.shape[0] != len(E):
            raise ValueError("X, T, and E must have same number of samples")
        
        # Store feature names
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        self.feature_names = feature_names
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create DataFrame for lifelines
        df = pd.DataFrame(X_scaled, columns=feature_names)
        df['T'] = T
        df['E'] = E
        
        # Fit model
        self.model.fit(df, duration_col='T', event_col='E')
        
        # Calculate concordance index
        self.concordance_index_ = self.model.concordance_index_
        
        # Extract feature importances (hazard ratios)
        self.feature_importances = np.exp(self.model.params_).values
        
        self.is_fitted = True
        return self
    
    def predict_risk_score(self, X):
        """
        Predict risk scores for new patients.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of risk scores (higher = higher risk)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Standardize features using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Get risk scores
        if self.model_type == 'cph':
            risk_scores = self.model.predict_partial_hazard(
                pd.DataFrame(X_scaled, columns=self.feature_names)
            ).values
        else:
            risk_scores = self.model.predict_expectation(
                pd.DataFrame(X_scaled, columns=self.feature_names)
            ).values
        
        return risk_scores
    
    def predict_survival_function(self, X):
        """
        Predict survival probability functions.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Survival probability curves for each patient
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        df = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        if self.model_type == 'cph':
            return self.model.predict_survival_function(df)
        else:
            return self.model.predict_survival_function(df)
    
    def get_risk_group(self, risk_scores, percentiles=(33, 67)):
        """
        Classify patients into risk groups.
        
        Args:
            risk_scores: Array of risk scores
            percentiles: Risk group boundaries (e.g., (33, 67) for low/med/high)
            
        Returns:
            Array of risk group labels ('LOW', 'INTERMEDIATE', 'HIGH')
        """
        p33, p67 = np.percentile(risk_scores, percentiles)
        
        risk_groups = np.where(
            risk_scores < p33, 'LOW',
            np.where(risk_scores < p67, 'INTERMEDIATE', 'HIGH')
        )
        
        return risk_groups
    
    def feature_selection(self, threshold=0.01):
        """
        Select important features based on coefficient magnitude.
        
        Args:
            threshold: Minimum absolute coefficient value
            
        Returns:
            Dictionary of selected features and their coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before feature selection")
        
        coeffs = self.model.params_
        important_features = {}
        
        for feature, coeff in zip(self.feature_names, coeffs):
            if abs(coeff) >= threshold:
                important_features[feature] = float(coeff)
        
        return important_features
    
    def cross_validate(self, T, E, X, cv=5):
        """
        Perform k-fold cross-validation.
        
        Args:
            T: Survival times
            E: Event indicators
            X: Features
            cv: Number of folds
            
        Returns:
            Dictionary with cross-validation results
        """
        scores = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            E_train, E_test = E[train_idx], E[test_idx]
            
            # Create and fit temporary model
            if self.model_type == 'cph':
                temp_model = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
            else:
                temp_model = WeibullAFTFitter(penalizer=self.penalizer)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            df_train = pd.DataFrame(X_train_scaled, columns=self.feature_names)
            df_train['T'] = T_train
            df_train['E'] = E_train
            
            temp_model.fit(df_train, duration_col='T', event_col='E')
            
            df_test = pd.DataFrame(X_test_scaled, columns=self.feature_names)
            c_index = concordance_index(T_test, temp_model.predict_partial_hazard(df_test), E_test)
            scores.append(c_index)
        
        return {
            'mean_c_index': np.mean(scores),
            'std_c_index': np.std(scores),
            'scores': scores
        }
    
    def get_summary(self):
        """Get model summary."""
        if not self.is_fitted:
            return "Model not yet fitted"
        
        summary = {
            'model_type': self.model_type,
            'penalizer': self.penalizer,
            'l1_ratio': self.l1_ratio,
            'concordance_index': float(self.concordance_index_),
            'features': self.feature_names,
            'coefficients': self.model.params_.to_dict(),
            'hazard_ratios': {feat: float(hr) for feat, hr in zip(self.feature_names, self.feature_importances)}
        }
        return summary
    
    def save(self, filepath):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model


class CoxNetEnsemble:
    """Ensemble of multiple Cox models for robust risk prediction."""
    
    def __init__(self, n_models=5, penalizers=None):
        """
        Initialize ensemble.
        
        Args:
            n_models: Number of models in ensemble
            penalizers: List of penalizer values for each model
        """
        self.n_models = n_models
        if penalizers is None:
            penalizers = np.logspace(-2, 1, n_models)  # 0.01 to 10
        self.penalizers = penalizers
        self.models = [CoxNetModel(penalizer=p) for p in penalizers]
    
    def fit(self, T, E, X, feature_names=None):
        """Fit all models in ensemble."""
        for model in self.models:
            model.fit(T, E, X, feature_names)
        return self
    
    def predict_risk_score(self, X):
        """Average risk scores across ensemble."""
        scores = np.array([model.predict_risk_score(X) for model in self.models])
        return np.mean(scores, axis=0)
    
    def predict_survival_function(self, X):
        """Average survival functions across ensemble."""
        surv_funcs = [model.predict_survival_function(X) for model in self.models]
        # Combine survival curves
        combined = surv_funcs[0].copy()
        for sf in surv_funcs[1:]:
            combined = combined.add(sf)
        return combined / len(surv_funcs)
    
    def get_risk_group(self, X):
        """Classify into risk groups using ensemble predictions."""
        risk_scores = self.predict_risk_score(X)
        return self.models[0].get_risk_group(risk_scores)


def prepare_survival_data(patients_df):
    """
    Prepare data for Cox model training.
    
    Args:
        patients_df: DataFrame with columns:
            - age: Patient age
            - wbc_count: White blood cell count
            - blast_percentage: Blast percentage
            - hemoglobin: Hemoglobin level
            - platelets: Platelet count
            - survival_days: Time to event
            - event_status: Event indicator (1=event, 0=censored)
            
    Returns:
        Tuple of (T, E, X, feature_names)
    """
    # Select features
    feature_cols = ['age', 'wbc_count', 'blast_percentage', 'hemoglobin', 'platelets']
    
    # Remove rows with missing values
    mask = ~patients_df[feature_cols + ['survival_days', 'event_status']].isnull().any(axis=1)
    clean_df = patients_df[mask].copy()
    
    # Extract survival data
    T = clean_df['survival_days'].values
    E = (clean_df['event_status'] != 'ALIVE').astype(int).values
    X = clean_df[feature_cols].values
    
    return T, E, X, feature_cols


def get_or_create_cox_model(model_path=None):
    """
    Get or create cached CoxNet model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        CoxNetModel instance
    """
    if model_path and os.path.exists(model_path):
        return CoxNetModel.load(model_path)
    
    return CoxNetModel(penalizer=0.1, l1_ratio=1.0)
