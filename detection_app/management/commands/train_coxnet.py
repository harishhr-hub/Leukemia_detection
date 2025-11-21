"""
Django management command to train CoxNet model from patient data.

Usage:
    python manage.py train_coxnet --output path/to/model.pkl --penalizers 0.01,0.1,1.0
"""

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import os
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Train CoxNet LASSO-Cox model from patient survival data'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--output',
            type=str,
            default='model/coxnet_model.pkl',
            help='Output path for trained model'
        )
        parser.add_argument(
            '--penalizers',
            type=str,
            default='0.01,0.1,1.0,10.0,100.0',
            help='Comma-separated penalizer values for ensemble'
        )
        parser.add_argument(
            '--test-split',
            type=float,
            default=0.2,
            help='Validation split fraction'
        )
        parser.add_argument(
            '--seed',
            type=int,
            default=42,
            help='Random seed'
        )
    
    def handle(self, *args, **options):
        try:
            from detection_app.models import Patient, PatientFollowUp
            from detection_app.coxnet_utils import CoxNetEnsemble
            
            self.stdout.write("Collecting patient data...")
            
            # Prepare data
            patients_data = []
            for patient in Patient.objects.all():
                try:
                    latest_followup = patient.follow_ups.latest('visit_date')
                    event_status = latest_followup.event_status
                    survival_days = latest_followup.survival_days
                    
                    patients_data.append({
                        'patient_id': patient.id,
                        'age': patient.age,
                        'wbc_count': patient.wbc_count or 0,
                        'blast_percentage': patient.blast_percentage or 0,
                        'hemoglobin': patient.hemoglobin or 0,
                        'platelets': patient.platelets or 0,
                        'survival_days': survival_days,
                        'event_status': event_status,
                    })
                except PatientFollowUp.DoesNotExist:
                    continue
            
            if not patients_data:
                raise CommandError('No patient follow-up data found. Please add patient data first.')
            
            df = pd.DataFrame(patients_data)
            self.stdout.write(f"Loaded {len(df)} patients")
            
            # Prepare features
            feature_cols = ['age', 'wbc_count', 'blast_percentage', 'hemoglobin', 'platelets']
            mask = ~df[feature_cols + ['survival_days', 'event_status']].isnull().any(axis=1)
            df_clean = df[mask].copy()
            
            if len(df_clean) == 0:
                raise CommandError('No complete records found after removing missing values.')
            
            X = df_clean[feature_cols].values
            T = df_clean['survival_days'].values
            E = (df_clean['event_status'] != 'ALIVE').astype(int).values
            
            self.stdout.write(f"Features: {feature_cols}")
            self.stdout.write(f"Training samples: {len(X)}, Events: {np.sum(E)}")
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_val, T_train, T_val, E_train, E_val = train_test_split(
                X, T, E,
                test_size=options['test_split'],
                random_state=options['seed']
            )
            
            self.stdout.write(f"Train: {len(X_train)}, Val: {len(X_val)}")
            
            # Parse penalizers
            penalizers = [float(p) for p in options['penalizers'].split(',')]
            self.stdout.write(f"Penalizers: {penalizers}")
            
            # Train ensemble
            self.stdout.write("Training CoxNet ensemble...")
            ensemble = CoxNetEnsemble(n_models=len(penalizers), penalizers=penalizers)
            ensemble.fit(T_train, E_train, X_train, feature_names=feature_cols)
            
            # Validate
            from lifelines.utils import concordance_index
            val_scores = ensemble.predict_risk_score(X_val)
            c_index = concordance_index(T_val, val_scores, E_val)
            
            self.stdout.write(self.style.SUCCESS(f"Validation C-index: {c_index:.4f}"))
            
            # Save
            output_path = os.path.join(settings.BASE_DIR, options['output'])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            ensemble.models[0].save(output_path)
            
            self.stdout.write(self.style.SUCCESS(f"Model saved to {output_path}"))
            
        except Exception as e:
            raise CommandError(f"Training failed: {str(e)}")
