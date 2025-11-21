"""
Training scripts for Mask R-CNN and CoxNet models

This module provides utilities for training both the Mask R-CNN cell detection model
and the CoxNet survival prediction model.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BloodSmearDataset(Dataset):
    """Dataset for blood smear images with annotations."""
    
    def __init__(self, image_dir, annotations_file, transforms=None):
        """
        Args:
            image_dir: Directory containing images
            annotations_file: JSON file with annotations
            transforms: Image transformations
        """
        self.image_dir = image_dir
        self.transforms = transforms
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_ids = list(self.annotations.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        
        # Get annotations (boxes and masks)
        ann = self.annotations[image_id]
        boxes = ann.get('boxes', [])
        masks = ann.get('masks', [])
        
        # Convert to tensors
        if self.transforms:
            image = self.transforms(image)
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'masks': torch.as_tensor(masks, dtype=torch.uint8),
            'image_id': torch.tensor([idx])
        }
        
        return image, target


def train_mask_rcnn(model, train_loader, val_loader, num_epochs=10, learning_rate=0.005,
                    save_path=None):
    """
    Train Mask R-CNN model.
    
    Args:
        model: MaskRCNNDetector instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        save_path: Path to save trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.to(device)
    
    # Optimizer
    params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.model.train()
        total_loss = 0
        
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
        # Validation
        if val_loader:
            val_loss = 0
            model.model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model.model(images, targets)
                    val_loss += sum(loss for loss in loss_dict.values()).item()
            
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"  Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }, save_path)
                    logger.info(f"  Saved best model to {save_path}")
        
        lr_scheduler.step()
    
    return model


def train_coxnet(X_train, T_train, E_train, X_val=None, T_val=None, E_val=None,
                 penalizers=None, feature_names=None, output_path=None):
    """
    Train CoxNet ensemble model.
    
    Args:
        X_train: Training features
        T_train: Training survival times
        E_train: Training event indicators
        X_val: Validation features
        T_val: Validation survival times
        E_val: Validation event indicators
        penalizers: List of penalizer values
        feature_names: Names of features
        output_path: Path to save trained model
        
    Returns:
        Trained CoxNetEnsemble model
    """
    from .coxnet_utils import CoxNetEnsemble
    
    if penalizers is None:
        penalizers = np.logspace(-2, 1, 5)  # 5 models with varying regularization
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    logger.info(f"Training CoxNet ensemble with {len(penalizers)} models")
    logger.info(f"Features: {feature_names}")
    logger.info(f"Training samples: {len(X_train)}, Events: {np.sum(E_train)}")
    
    # Create and train ensemble
    ensemble = CoxNetEnsemble(n_models=len(penalizers), penalizers=penalizers)
    ensemble.fit(T_train, E_train, X_train, feature_names)
    
    # Validate
    if X_val is not None:
        from lifelines.utils import concordance_index
        val_scores = ensemble.predict_risk_score(X_val)
        c_index = concordance_index(T_val, val_scores, E_val)
        logger.info(f"Validation C-index: {c_index:.4f}")
    
    # Save model
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ensemble.models[0].save(output_path)  # Save first model as representative
        logger.info(f"Model saved to {output_path}")
    
    return ensemble


def prepare_coxnet_data_from_django(save_path=None):
    """
    Prepare survival data from Django models.
    
    Args:
        save_path: Optional path to save the data
        
    Returns:
        Tuple of (X, T, E, feature_names, patients_df)
    """
    # This should be run within Django context
    try:
        from django.conf import settings
        import django
        
        if not settings.configured:
            django.setup()
        
        from detection_app.models import Patient, PatientFollowUp
        
        # Collect patient data
        patients_data = []
        
        for patient in Patient.objects.all():
            # Get follow-up data
            latest_followup = patient.follow_ups.latest('visit_date')
            
            if latest_followup:
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
        
        if not patients_data:
            logger.warning("No patient data found")
            return None, None, None, None, None
        
        df = pd.DataFrame(patients_data)
        
        # Prepare features
        feature_cols = ['age', 'wbc_count', 'blast_percentage', 'hemoglobin', 'platelets']
        
        # Remove rows with missing values
        mask = ~df[feature_cols + ['survival_days', 'event_status']].isnull().any(axis=1)
        df_clean = df[mask].copy()
        
        X = df_clean[feature_cols].values
        T = df_clean['survival_days'].values
        E = (df_clean['event_status'] != 'ALIVE').astype(int).values
        
        logger.info(f"Prepared data: {len(df_clean)} patients, {np.sum(E)} events")
        
        if save_path:
            df_clean.to_csv(save_path, index=False)
            logger.info(f"Data saved to {save_path}")
        
        return X, T, E, feature_cols, df_clean
    
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None, None, None, None, None


# Example usage and validation
if __name__ == "__main__":
    print("Mask R-CNN and CoxNet Training Module")
    print("=" * 50)
    print("\nUsage:")
    print("1. For Mask R-CNN training: Use train_mask_rcnn()")
    print("2. For CoxNet training: Use train_coxnet()")
    print("\nExample:")
    print("  X_train, T_train, E_train = prepare_data()")
    print("  ensemble = train_coxnet(X_train, T_train, E_train)")
