"""
Example Usage: Mask R-CNN and CoxNet Integration

This file demonstrates how to use the integrated models in various scenarios.
"""

# ==============================================================================
# EXAMPLE 1: Basic Cell Detection
# ==============================================================================

def example_basic_cell_detection():
    """Detect cells in a blood smear image."""
    from detection_app.mask_rcnn_utils import detect_cells_in_image
    
    image_path = 'media/smear_images/sample.jpg'
    
    # Detect cells
    results = detect_cells_in_image(image_path, confidence_threshold=0.5)
    
    # Print results
    print(f"Found {results['num_cells']} cells")
    print(f"Cell density: {results['cell_density']:.4f}")
    print(f"Average cell area: {sum(results['cell_statistics']['cell_areas']) / len(results['cell_statistics']['cell_areas']):.0f}")
    
    # Access bounding boxes
    for idx, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
        x1, y1, x2, y2 = box
        print(f"  Cell {idx+1}: Box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}), Confidence={score:.3f}")
    
    # Save annotated image
    from PIL import Image
    annotated = Image.fromarray(results['annotated_image'].astype('uint8'))
    annotated.save('media/annotated_result.jpg')


# ==============================================================================
# EXAMPLE 2: Create Detection Record in Django
# ==============================================================================

def example_create_detection_record():
    """Create a detection result with cell statistics."""
    from django.contrib.auth.models import User
    from detection_app.models import Patient, DetectionResult
    from detection_app.mask_rcnn_utils import detect_cells_in_image
    
    # Get patient
    user = User.objects.first()
    patient = Patient.objects.create(
        user=user,
        name='John Doe',
        age=45,
        gender='M'
    )
    
    # Detect cells
    results = detect_cells_in_image('media/smear_images/sample.jpg')
    
    # Create detection record
    detection = DetectionResult.objects.create(
        patient=patient,
        image='smear_images/sample.jpg',
        detection_result='Positive' if results['num_cells'] > 10 else 'Negative',
        risk_level='HIGH' if results['num_cells'] > 100 else 'LOW',
        cell_count=results['num_cells'],
        cell_density=results['cell_density'],
        cell_statistics=results['cell_statistics']
    )
    
    print(f"Created detection {detection.id} for patient {patient.name}")
    print(f"  Cells: {detection.cell_count}")
    print(f"  Risk: {detection.risk_level}")


# ==============================================================================
# EXAMPLE 3: Train CoxNet Model
# ==============================================================================

def example_train_coxnet_model():
    """Train CoxNet on patient survival data."""
    from detection_app.models import Patient, PatientFollowUp
    from detection_app.coxnet_utils import CoxNetEnsemble, prepare_survival_data
    from django.contrib.auth.models import User
    from datetime import date, timedelta
    import numpy as np
    
    # Create sample patients with follow-up data
    user = User.objects.first()
    
    for i in range(50):
        patient = Patient.objects.create(
            user=user,
            name=f'Patient {i}',
            age=30 + np.random.randint(0, 50),
            gender=np.random.choice(['M', 'F']),
            wbc_count=np.random.uniform(5, 50),
            blast_percentage=np.random.uniform(5, 95),
            hemoglobin=np.random.uniform(7, 14),
            platelets=np.random.uniform(10, 300)
        )
        
        # Add follow-up
        PatientFollowUp.objects.create(
            patient=patient,
            visit_date=date.today() - timedelta(days=np.random.randint(30, 730)),
            event_status=np.random.choice(['ALIVE', 'DEATH', 'RELAPSE']),
            survival_days=np.random.randint(30, 730),
            notes='Sample follow-up'
        )
    
    # Prepare data
    X, T, E, feature_cols = prepare_survival_data(Patient.objects.all())
    
    if X is None:
        print("Not enough data to train")
        return
    
    # Create ensemble
    penalizers = [0.01, 0.1, 1.0, 10.0, 100.0]
    ensemble = CoxNetEnsemble(n_models=len(penalizers), penalizers=penalizers)
    
    # Train
    print(f"Training CoxNet ensemble on {len(X)} patients with {np.sum(E)} events")
    ensemble.fit(T, E, X, feature_names=feature_cols)
    
    # Cross-validate
    cv_results = ensemble.models[0].cross_validate(T, E, X, cv=5)
    print(f"Cross-validation C-index: {cv_results['mean_c_index']:.4f} ± {cv_results['std_c_index']:.4f}")
    
    # Save
    ensemble.models[0].save('model/coxnet_model.pkl')
    print("Model saved to model/coxnet_model.pkl")


# ==============================================================================
# EXAMPLE 4: Predict Risk Score for Patient
# ==============================================================================

def example_predict_risk_score():
    """Predict survival risk for a patient."""
    from detection_app.models import Patient, SurvivalAnalysis
    from detection_app.coxnet_utils import get_or_create_cox_model
    import numpy as np
    
    # Get patient
    patient = Patient.objects.first()
    
    # Load model
    model = get_or_create_cox_model('model/coxnet_model.pkl')
    
    if not model.is_fitted:
        print("Model not trained yet. Run example_train_coxnet_model() first.")
        return
    
    # Prepare features
    patient_features = np.array([[
        patient.age,
        patient.wbc_count or 0,
        patient.blast_percentage or 0,
        patient.hemoglobin or 0,
        patient.platelets or 0
    ]])
    
    # Predict
    risk_score = model.predict_risk_score(patient_features)[0]
    risk_group = model.get_risk_group(np.array([risk_score]))[0]
    
    # Get survival function
    survival_func = model.predict_survival_function(patient_features)
    
    # Extract probabilities at key timepoints
    print(f"\nRisk Prediction for {patient.name}:")
    print(f"  Risk Score: {risk_score:.4f}")
    print(f"  Risk Group: {risk_group}")
    
    try:
        prob_12m = survival_func.iloc[:, 0].loc[365]
        print(f"  12-month Survival: {prob_12m:.1%}")
    except:
        print("  12-month Survival: N/A")
    
    # Save to database
    analysis, _ = SurvivalAnalysis.objects.get_or_create(patient=patient)
    analysis.risk_score = risk_score
    analysis.risk_group = risk_group
    analysis.model_features = dict(zip(model.feature_names, model.feature_importances.tolist()))
    analysis.save()
    
    print(f"\nSaved analysis to database")


# ==============================================================================
# EXAMPLE 5: Feature Importance and Selection
# ==============================================================================

def example_feature_importance():
    """Extract and interpret feature importance."""
    from detection_app.coxnet_utils import get_or_create_cox_model
    
    model = get_or_create_cox_model('model/coxnet_model.pkl')
    
    if not model.is_fitted:
        print("Model not trained yet.")
        return
    
    # Get feature importances (hazard ratios)
    print("\nFeature Importance (Hazard Ratios):")
    print("=" * 50)
    
    summary = model.get_summary()
    hr = summary['hazard_ratios']
    
    for feature, ratio in sorted(hr.items(), key=lambda x: abs(x[1] - 1), reverse=True):
        pct_change = (ratio - 1) * 100
        direction = "↑ increases risk" if ratio > 1 else "↓ decreases risk"
        print(f"{feature:20s}: {ratio:6.3f} ({pct_change:+6.1f}%) {direction}")
    
    # Feature selection
    print("\nSelected Features (|coefficient| >= 0.01):")
    print("=" * 50)
    
    selected = model.feature_selection(threshold=0.01)
    for feature, coeff in selected.items():
        print(f"  {feature:20s}: {coeff:7.4f}")


# ==============================================================================
# EXAMPLE 6: Batch Processing Multiple Images
# ==============================================================================

def example_batch_processing():
    """Process multiple blood smear images."""
    from detection_app.models import Patient, DetectionResult
    from detection_app.mask_rcnn_utils import detect_cells_in_image
    import glob
    
    patient = Patient.objects.first()
    image_files = glob.glob('media/smear_images/*.jpg')
    
    print(f"Processing {len(image_files)} images...")
    
    for idx, image_path in enumerate(image_files, 1):
        try:
            results = detect_cells_in_image(image_path)
            
            detection = DetectionResult.objects.create(
                patient=patient,
                image=image_path,
                detection_result='Positive' if results['num_cells'] > 10 else 'Negative',
                risk_level='HIGH' if results['num_cells'] > 100 else 'LOW',
                cell_count=results['num_cells'],
                cell_density=results['cell_density'],
                cell_statistics=results['cell_statistics']
            )
            
            print(f"  [{idx}] {image_path}: {results['num_cells']} cells - {detection.risk_level} risk")
        
        except Exception as e:
            print(f"  [{idx}] {image_path}: ERROR - {str(e)}")
    
    print(f"Processed {len(image_files)} images")


# ==============================================================================
# EXAMPLE 7: Cross-Validation and Model Evaluation
# ==============================================================================

def example_cross_validation():
    """Perform cross-validation on CoxNet model."""
    from detection_app.models import Patient
    from detection_app.coxnet_utils import CoxNetModel, prepare_survival_data
    import numpy as np
    
    # Prepare data
    X, T, E, feature_cols = prepare_survival_data(Patient.objects.all())
    
    if X is None or len(X) < 20:
        print("Insufficient data for cross-validation")
        return
    
    # Create model
    model = CoxNetModel(penalizer=0.1, l1_ratio=1.0)
    
    # 5-fold cross-validation
    print("Performing 5-fold cross-validation...")
    cv_results = model.cross_validate(T, E, X, cv=5)
    
    print(f"\nCross-Validation Results:")
    print(f"  Mean C-index: {cv_results['mean_c_index']:.4f}")
    print(f"  Std C-index:  {cv_results['std_c_index']:.4f}")
    print(f"  Individual folds:")
    
    for i, score in enumerate(cv_results['scores'], 1):
        print(f"    Fold {i}: {score:.4f}")


# ==============================================================================
# EXAMPLE 8: Generate Predictions with Confidence
# ==============================================================================

def example_predictions_with_confidence():
    """Generate predictions and confidence intervals."""
    from detection_app.models import Patient
    from detection_app.coxnet_utils import get_or_create_cox_model
    import numpy as np
    
    model = get_or_create_cox_model('model/coxnet_model.pkl')
    
    if not model.is_fitted:
        print("Model not trained")
        return
    
    # Get all patients
    patients = Patient.objects.all()[:10]
    
    print(f"Predictions for {len(patients)} patients:")
    print("=" * 70)
    print(f"{'Name':<20} {'Age':>5} {'Risk Score':>12} {'Risk Group':<15} {'1-Year Prob':>12}")
    print("-" * 70)
    
    for patient in patients:
        features = np.array([[
            patient.age,
            patient.wbc_count or 0,
            patient.blast_percentage or 0,
            patient.hemoglobin or 0,
            patient.platelets or 0
        ]])
        
        risk_score = model.predict_risk_score(features)[0]
        risk_group = model.get_risk_group(np.array([risk_score]))[0]
        
        try:
            surv_func = model.predict_survival_function(features)
            prob_1yr = surv_func.iloc[:, 0].loc[365]
        except:
            prob_1yr = np.nan
        
        print(f"{patient.name:<20} {patient.age:>5} {risk_score:>12.4f} {risk_group:<15} {prob_1yr:>12.1%}")


# ==============================================================================
# MAIN: Run Examples
# ==============================================================================

if __name__ == '__main__':
    print("Mask R-CNN and CoxNet Examples")
    print("=" * 70)
    
    # Uncomment examples to run:
    
    # example_basic_cell_detection()
    # example_create_detection_record()
    # example_train_coxnet_model()
    # example_predict_risk_score()
    # example_feature_importance()
    # example_batch_processing()
    # example_cross_validation()
    # example_predictions_with_confidence()
    
    print("\nTo run examples, uncomment the desired function calls above")
    print("and execute: python manage.py shell < examples.py")
