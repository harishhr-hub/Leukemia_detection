"""
Integrated detection views using Mask R-CNN and CoxNet

These views combine cell detection with risk prediction for comprehensive analysis.
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from PIL import Image
import numpy as np
import json
import os
from io import BytesIO
import base64
from django.conf import settings

from .models import Patient, DetectionResult, SurvivalAnalysis
from .mask_rcnn_utils import get_mask_rcnn_detector
from .coxnet_utils import get_or_create_cox_model, prepare_survival_data


@login_required
def detect_with_mask_rcnn(request, patient_id):
    """
    Advanced detection using Mask R-CNN for cell-level analysis.
    
    Performs:
    1. Cell detection and segmentation
    2. Feature extraction (cell count, density, statistics)
    3. Risk assessment
    """
    patient = get_object_or_404(Patient, id=patient_id, user=request.user)
    
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Get detector
            detector = get_mask_rcnn_detector()
            
            # Save uploaded image temporarily
            uploaded_file = request.FILES['image']
            temp_path = os.path.join(settings.MEDIA_ROOT, 'temp_detection.jpg')
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            
            with open(temp_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            
            # Run detection
            detection_results = detector.detect(temp_path, confidence_threshold=0.5)
            
            # Create detection record
            detection = DetectionResult.objects.create(
                patient=patient,
                image=uploaded_file,
                detection_result='Positive' if detection_results['num_cells'] > 0 else 'Negative',
                risk_level=_assign_risk_level(detection_results),
                cell_count=detection_results['num_cells'],
                cell_density=detection_results['cell_density'],
                cell_statistics=detection_results['cell_statistics'],
            )
            
            # Save annotated image
            annotated_img = Image.fromarray(detection_results['annotated_image'].astype(np.uint8))
            img_bytes = BytesIO()
            annotated_img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            from django.core.files.base import ContentFile
            detection.annotated_image.save(
                f'detection_{detection.id}_annotated.jpg',
                ContentFile(img_bytes.read()),
                save=True
            )
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            messages.success(request, f'Detection completed. Found {detection_results["num_cells"]} cells.')
            return redirect('patient_detail', pk=patient.id)
        
        except Exception as e:
            messages.error(request, f'Detection failed: {str(e)}')
            return redirect('patient_detail', pk=patient.id)
    
    return render(request, 'detection_app/detect_mask_rcnn.html', {'patient': patient})


@login_required
def predict_survival_risk(request, patient_id):
    """
    Predict survival risk using CoxNet model.
    
    Uses patient clinical features to compute:
    - Risk score
    - Risk group classification
    - Survival probability curves
    """
    patient = get_object_or_404(Patient, id=patient_id, user=request.user)
    
    try:
        # Check if patient has required clinical features
        required_fields = ['age', 'wbc_count', 'blast_percentage', 'hemoglobin', 'platelets']
        if not all(hasattr(patient, field) and getattr(patient, field) is not None for field in required_fields):
            messages.warning(request, 'Patient missing clinical features for risk prediction.')
            return redirect('patient_detail', pk=patient.id)
        
        # Load or create Cox model
        model_path = os.path.join(settings.BASE_DIR, 'model', 'coxnet_model.pkl')
        cox_model = get_or_create_cox_model(model_path)
        
        # If model not trained, train it
        if not cox_model.is_fitted:
            X, T, E, feature_names = prepare_survival_data_from_django()
            if X is None:
                messages.error(request, 'Insufficient data for model training.')
                return redirect('patient_detail', pk=patient.id)
            
            cox_model.fit(T, E, X, feature_names)
        
        # Prepare patient features
        patient_features = np.array([[
            patient.age,
            patient.wbc_count or 0,
            patient.blast_percentage or 0,
            patient.hemoglobin or 0,
            patient.platelets or 0
        ]])
        
        # Predict
        risk_score = cox_model.predict_risk_score(patient_features)[0]
        risk_group = cox_model.get_risk_group(np.array([risk_score]))[0]
        
        # Get survival function
        survival_func = cox_model.predict_survival_function(patient_features)
        
        # Extract survival probabilities at key timepoints
        times = [365, 730, 1825]  # 1, 2, 5 years
        survival_probs = {}
        for t in times:
            try:
                prob = float(survival_func.iloc[:, 0].loc[survival_func.index.get_indexer([t], method='nearest')[0]])
                months = t // 30
                survival_probs[f'{months}m_survival'] = prob
            except:
                survival_probs[f'{t}_days_survival'] = None
        
        # Save or update survival analysis
        survival_analysis, created = SurvivalAnalysis.objects.get_or_create(patient=patient)
        survival_analysis.risk_score = risk_score
        survival_analysis.risk_group = risk_group
        survival_analysis.survival_probability_12m = survival_probs.get('12m_survival')
        survival_analysis.survival_probability_24m = survival_probs.get('24m_survival')
        survival_analysis.survival_probability_60m = survival_probs.get('60m_survival')
        survival_analysis.model_features = dict(zip(cox_model.feature_names, cox_model.feature_importances.tolist()))
        survival_analysis.hazard_ratios = cox_model.get_summary()['hazard_ratios']
        survival_analysis.concordance_index = cox_model.concordance_index_
        survival_analysis.save()
        
        messages.success(request, 'Survival risk prediction completed.')
        return redirect('patient_detail', pk=patient.id)
    
    except Exception as e:
        messages.error(request, f'Risk prediction failed: {str(e)}')
        return redirect('patient_detail', pk=patient.id)


@login_required
def view_cell_detection_details(request, detection_id):
    """
    View detailed cell detection results with visualizations.
    """
    detection = get_object_or_404(DetectionResult, id=detection_id, patient__user=request.user)
    
    context = {
        'detection': detection,
        'cell_count': detection.cell_count,
        'cell_density': detection.cell_density,
        'cell_statistics': detection.cell_statistics,
        'has_masks': detection.detection_masks is not None,
    }
    
    return render(request, 'detection_app/cell_detection_details.html', context)


@login_required
def view_survival_risk(request, patient_id):
    """
    View survival risk analysis results with survival curves.
    """
    patient = get_object_or_404(Patient, id=patient_id, user=request.user)
    survival_analysis = getattr(patient, 'survival_analysis', None)
    
    if not survival_analysis:
        messages.info(request, 'No survival analysis available for this patient.')
        return redirect('patient_detail', pk=patient.id)
    
    context = {
        'patient': patient,
        'survival_analysis': survival_analysis,
        'risk_color': _get_risk_color(survival_analysis.risk_group),
    }
    
    return render(request, 'detection_app/survival_risk_detail.html', context)


# Helper functions

def _assign_risk_level(detection_results):
    """Assign risk level based on cell detection results."""
    num_cells = detection_results['num_cells']
    cell_density = detection_results['cell_density']
    
    if num_cells == 0:
        return 'LOW'
    elif num_cells > 100 or cell_density > 0.5:
        return 'HIGH'
    elif num_cells > 50:
        return 'MEDIUM'
    else:
        return 'LOW'


def _get_risk_color(risk_group):
    """Get color for risk group visualization."""
    colors = {
        'LOW': '#28a745',
        'INTERMEDIATE': '#ffc107',
        'HIGH': '#dc3545'
    }
    return colors.get(risk_group, '#6c757d')


def prepare_survival_data_from_django():
    """Prepare survival data from Django models."""
    from .models import Patient, PatientFollowUp
    
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
        return None, None, None, None
    
    import pandas as pd
    df = pd.DataFrame(patients_data)
    
    feature_cols = ['age', 'wbc_count', 'blast_percentage', 'hemoglobin', 'platelets']
    mask = ~df[feature_cols + ['survival_days', 'event_status']].isnull().any(axis=1)
    df_clean = df[mask].copy()
    
    X = df_clean[feature_cols].values
    T = df_clean['survival_days'].values
    E = (df_clean['event_status'] != 'ALIVE').astype(int).values
    
    return X, T, E, feature_cols
