from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Patient(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=10, choices=[
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other')
    ])
    medical_history = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    
    # Clinical features for survival analysis
    wbc_count = models.FloatField(null=True, blank=True, help_text="White blood cell count (K/µL)")
    blast_percentage = models.FloatField(null=True, blank=True, help_text="Blast percentage (%)")
    hemoglobin = models.FloatField(null=True, blank=True, help_text="Hemoglobin (g/dL)")
    platelets = models.FloatField(null=True, blank=True, help_text="Platelet count (K/µL)")
    cytogenetics = models.CharField(max_length=100, blank=True, help_text="Cytogenetic risk (e.g., t(9;22), normal, etc.)")
    risk_category = models.CharField(max_length=20, choices=[
        ('STANDARD', 'Standard Risk'),
        ('HIGH', 'High Risk'),
        ('UNKNOWN', 'Unknown')
    ], default='UNKNOWN', blank=True)

    def __str__(self):
        return f"{self.name} - {self.age} years"

class DetectionResult(models.Model):
    # Leukemia type choices
    LEUKEMIA_TYPES = [
        ('ALL', 'Acute Lymphoblastic Leukemia'),
        ('AML', 'Acute Myeloid Leukemia'),
        ('CLL', 'Chronic Lymphocytic Leukemia'),
        ('CML', 'Chronic Myeloid Leukemia'),
        ('NORMAL', 'Normal/Healthy')
    ]
    
    # ALL FAB classifications
    ALL_FAB_CHOICES = [
        ('L1', 'L1 - Small uniform cells'),
        ('L2', 'L2 - Large heterogeneous cells'),
        ('L3', 'L3 - Large, heavily vacuolated cells (Burkitt-like)'),
    ]
    
    # AML FAB classifications
    AML_FAB_CHOICES = [
        ('M0', 'M0 - Undifferentiated AML'),
        ('M1', 'M1 - AML with minimal differentiation'),
        ('M2', 'M2 - AML with differentiation'),
        ('M3', 'M3 - Acute Promyelocytic Leukemia (APL)'),
        ('M4', 'M4 - Acute Myelomonocytic Leukemia'),
        ('M5', 'M5 - Acute Monocytic Leukemia'),
        ('M6', 'M6 - Acute Erythroid Leukemia'),
        ('M7', 'M7 - Acute Megakaryocytic Leukemia'),
    ]
    
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='detections')
    image = models.ImageField(upload_to='smear_images/')
    detection_result = models.CharField(max_length=50)
    risk_level = models.CharField(max_length=20, choices=[
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High')
    ])
    date_detected = models.DateTimeField(default=timezone.now)
    report_generated = models.BooleanField(default=False)
    report_file = models.FileField(upload_to='reports/', null=True, blank=True)
    
    # Leukemia classification
    leukemia_type = models.CharField(max_length=20, choices=LEUKEMIA_TYPES, null=True, blank=True, 
                                      help_text="Detected leukemia type (ALL, AML, CLL, CML, or Normal)")
    all_fab_classification = models.CharField(max_length=10, choices=ALL_FAB_CHOICES, null=True, blank=True,
                                              help_text="FAB classification for ALL (L1, L2, L3)")
    aml_fab_classification = models.CharField(max_length=10, choices=AML_FAB_CHOICES, null=True, blank=True,
                                              help_text="FAB classification for AML (M0-M7)")
    fab_confidence = models.FloatField(null=True, blank=True, help_text="Confidence score for FAB classification (0-1)")
    
    # Mask R-CNN cell detection fields
    cell_count = models.IntegerField(null=True, blank=True, help_text="Number of detected cells")
    cell_density = models.FloatField(null=True, blank=True, help_text="Cell density (cells per million pixels)")
    cell_statistics = models.JSONField(null=True, blank=True, help_text="Cell area, intensity, circularity statistics")
    annotated_image = models.ImageField(upload_to='annotated_images/', null=True, blank=True, help_text="Image with detection boxes")
    detection_masks = models.JSONField(null=True, blank=True, help_text="Cell segmentation masks")

    def __str__(self):
        return f"Detection for {self.patient.name} - {self.date_detected}"

    class Meta:
        ordering = ['-date_detected']


class PatientFollowUp(models.Model):
    """Track patient follow-up visits and survival outcomes."""
    EVENT_CHOICES = [
        ('ALIVE', 'Alive (censored)'),
        ('RELAPSE', 'Relapsed'),
        ('DEATH', 'Death'),
    ]
    
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='follow_ups')
    visit_date = models.DateField()
    event_status = models.CharField(max_length=20, choices=EVENT_CHOICES, default='ALIVE')
    survival_days = models.IntegerField(help_text="Days from diagnosis to event or last follow-up")
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Follow-up: {self.patient.name} on {self.visit_date}"

    class Meta:
        ordering = ['-visit_date']
        unique_together = ('patient', 'visit_date')


class SurvivalAnalysis(models.Model):
    """Store Cox regression model results and risk scores."""
    patient = models.OneToOneField(Patient, on_delete=models.CASCADE, related_name='survival_analysis')
    risk_score = models.FloatField(null=True, blank=True, help_text="Computed risk score from Cox model")
    risk_group = models.CharField(max_length=20, choices=[
        ('LOW', 'Low Risk'),
        ('INTERMEDIATE', 'Intermediate Risk'),
        ('HIGH', 'High Risk')
    ], null=True, blank=True)
    predicted_survival_months = models.FloatField(null=True, blank=True, help_text="Predicted median survival in months")
    model_features = models.JSONField(null=True, blank=True, help_text="Dictionary of features used and their coefficients")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    # CoxNet specific fields
    concordance_index = models.FloatField(null=True, blank=True, help_text="Model C-index (discrimination ability)")
    hazard_ratios = models.JSONField(null=True, blank=True, help_text="Hazard ratios for each feature")
    survival_probability_12m = models.FloatField(null=True, blank=True, help_text="12-month survival probability")
    survival_probability_24m = models.FloatField(null=True, blank=True, help_text="24-month survival probability")
    survival_probability_60m = models.FloatField(null=True, blank=True, help_text="60-month survival probability")
    model_type = models.CharField(max_length=20, default='cph', choices=[
        ('cph', 'Cox Proportional Hazards'),
        ('weibull', 'Weibull AFT'),
        ('loglogistic', 'Log-Logistic AFT')
    ])
    penalizer_value = models.FloatField(null=True, blank=True, help_text="LASSO penalizer (lambda)")
    l1_ratio = models.FloatField(default=1.0, help_text="L1/L2 ratio for elastic net (1.0=LASSO)")

    def __str__(self):
        return f"Survival Analysis: {self.patient.name}"