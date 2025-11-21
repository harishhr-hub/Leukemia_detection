from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Patient, DetectionResult
from .forms import CustomUserCreationForm, PatientForm, DetectionForm
from .cnn_model import load_or_create_cnn_model, predict_with_cnn
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
from reportlab.pdfgen import canvas
from django.http import FileResponse
import os
from django.conf import settings
import uuid
import time

# Cache the model to avoid repeated loads
MODEL = None
MODEL_INFO = {
    'path': os.path.join(str(settings.BASE_DIR), 'model', 'cnn_detection_model.h5'),
    'input_size': (224, 224)
}

def load_model_once():
    global MODEL
    if MODEL is not None:
        return MODEL
    model_path = MODEL_INFO['path']
    MODEL = load_or_create_cnn_model(model_path, input_size=(224, 224), num_classes=2)
    return MODEL

def home(request):
    return render(request, 'detection_app/home.html')

def signup(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Account created successfully. Please log in.')
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'detection_app/signup.html', {'form': form})

@login_required
def dashboard(request):
    patients = Patient.objects.filter(user=request.user)
    return render(request, 'detection_app/dashboard.html', {'patients': patients})

@login_required
def add_patient(request):
    if request.method == 'POST':
        form = PatientForm(request.POST)
        if form.is_valid():
            patient = form.save(commit=False)
            patient.user = request.user
            patient.save()
            messages.success(request, 'Patient added successfully.')
            return redirect('dashboard')
    else:
        form = PatientForm()
    return render(request, 'detection_app/add_patient.html', {'form': form})

@login_required
def patient_detail(request, pk):
    patient = get_object_or_404(Patient, pk=pk, user=request.user)
    detections = patient.detections.all()
    return render(request, 'detection_app/patient_detail.html', {
        'patient': patient,
        'detections': detections
    })

@login_required
def detect_leukemia(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id, user=request.user)
    
    if request.method == 'POST':
        form = DetectionForm(request.POST, request.FILES)
        if form.is_valid():
            detection = form.save(commit=False)
            detection.patient = patient
            
            # --- Improved preprocessing and prediction ---
            # Load model once
            model = load_model_once()
            if model is None:
                messages.error(request, 'Detection model not available. Please contact the administrator.')
                return redirect('patient_detail', pk=patient.id)

            # Get prediction using CNN model
            image_path = request.FILES['image']
            detection_result = predict_with_cnn(model, image_path, input_size=(224, 224))
            cnn_probability = detection_result['probability']
            
            # Always perform FAB classification to verify cell type
            from detection_app.fab_classification import FABClassifier
            image_file = request.FILES['image']
            image_file.seek(0)
            
            fab_results = FABClassifier.detect_fab_classification(image_file)
            detection.leukemia_type = fab_results['leukemia_type']
            detection.fab_confidence = fab_results['fab_confidence']
            
            # Double verification: Only show Positive if BOTH conditions are met:
            # 1. CNN confidence is high (>= 0.70)
            # 2. FAB classifier identifies actual leukemia (not NORMAL)
            if cnn_probability >= 0.70 and detection.leukemia_type != 'NORMAL':
                # Confirmed leukemic cells
                detection.detection_result = 'Positive'
                if detection.leukemia_type in ['ALL', 'AML']:
                    detection.risk_level = 'HIGH'
                elif detection.leukemia_type in ['CLL', 'CML']:
                    detection.risk_level = 'MEDIUM'
                else:
                    detection.risk_level = 'LOW'
            else:
                # Not leukemic (either low CNN confidence OR FAB says NORMAL)
                detection.detection_result = 'Negative'
                detection.leukemia_type = 'NORMAL'
                detection.fab_confidence = None
                detection.risk_level = 'LOW'
            
            # Set ALL/AML specific FAB classifications only if confirmed leukemic
            if detection.detection_result == 'Positive':
                if fab_results['leukemia_type'] == 'ALL':
                    if fab_results['fab_classification']:
                        detection.all_fab_classification = fab_results['fab_classification']
                elif fab_results['leukemia_type'] == 'AML':
                    if fab_results['fab_classification']:
                        detection.aml_fab_classification = fab_results['fab_classification']
            
            # Save uploaded image with unique filename
            from django.core.files.base import ContentFile
            image_file = request.FILES['image']
            # Create unique filename using timestamp and uuid
            ext = image_file.name.split('.')[-1] if '.' in image_file.name else 'jpg'
            unique_filename = f'smear_{int(time.time())}_{uuid.uuid4().hex[:8]}.{ext}'
            
            # Reset file pointer to beginning
            image_file.seek(0)
            detection.image.save(unique_filename, ContentFile(image_file.read()), save=False)
            
            # Now save the detection record
            detection.save()
            messages.success(request, 'Detection completed successfully.')
            return redirect('patient_detail', pk=patient.id)
    else:
        form = DetectionForm()
    
    return render(request, 'detection_app/detect_leukemia.html', {
        'form': form,
        'patient': patient
    })

@login_required
def generate_report(request, detection_id):
    detection = get_object_or_404(DetectionResult, id=detection_id, patient__user=request.user)
    
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.units import inch
    from datetime import datetime
    
    # Create PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=8,
        spaceBefore=8
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4
    )
    
    # Title
    story.append(Paragraph("LEUKEMIA DETECTION REPORT", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Patient Information Section with border
    patient_data = [
        ['PATIENT INFORMATION', ''],
        ['Patient Name:', detection.patient.name],
        ['Age:', str(detection.patient.age)],
        ['Gender:', detection.patient.get_gender_display()],
        ['Date of Detection:', detection.date_detected.strftime('%d-%m-%Y %H:%M')],
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Detection Results Section
    detection_data = [
        ['DETECTION RESULTS', ''],
        ['Detection Result:', detection.detection_result],
        ['Risk Level:', detection.risk_level],
    ]
    
    # Add leukemia type if available
    if detection.leukemia_type:
        leukemia_display = {
            'ALL': 'Acute Lymphoblastic Leukemia (ALL)',
            'AML': 'Acute Myeloid Leukemia (AML)',
            'CLL': 'Chronic Lymphocytic Leukemia (CLL)',
            'CML': 'Chronic Myeloid Leukemia (CML)',
            'NORMAL': 'Normal/Healthy Blood'
        }
        detection_data.append(['Leukemia Type:', leukemia_display.get(detection.leukemia_type, detection.leukemia_type)])
    
    # Add FAB classification if available
    if detection.all_fab_classification:
        all_fab_labels = {
            'L1': 'L1 - Small uniform cells',
            'L2': 'L2 - Large heterogeneous cells',
            'L3': 'L3 - Large, heavily vacuolated cells (Burkitt-like)',
        }
        fab_text = all_fab_labels.get(detection.all_fab_classification, detection.all_fab_classification)
        detection_data.append(['ALL FAB Classification:', fab_text])
    
    if detection.aml_fab_classification:
        aml_fab_labels = {
            'M0': 'M0 - Undifferentiated AML',
            'M1': 'M1 - AML with minimal differentiation',
            'M2': 'M2 - AML with differentiation',
            'M3': 'M3 - Acute Promyelocytic Leukemia (APL)',
            'M4': 'M4 - Acute Myelomonocytic Leukemia',
            'M5': 'M5 - Acute Monocytic Leukemia',
            'M6': 'M6 - Acute Erythroid Leukemia',
            'M7': 'M7 - Acute Megakaryocytic Leukemia',
        }
        fab_text = aml_fab_labels.get(detection.aml_fab_classification, detection.aml_fab_classification)
        detection_data.append(['AML FAB Classification:', fab_text])
    
    detection_table = Table(detection_data, colWidths=[2*inch, 4*inch])
    detection_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('WORDWRAP', (1, 1), (1, -1), True),
    ]))
    story.append(detection_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Recommendations Section
    recommendations = {
        'LOW': [
            "Regular check-ups every 6 months",
            "Maintain healthy lifestyle",
            "Regular blood tests",
            "Monitor for any symptoms"
        ],
        'MEDIUM': [
            "Monthly check-ups recommended",
            "Consultation with hematologist",
            "Regular blood count monitoring",
            "Consider bone marrow examination",
            "Follow prescribed medication schedule"
        ],
        'HIGH': [
            "Immediate specialist consultation required",
            "Comprehensive treatment plan needed",
            "Weekly monitoring of blood counts",
            "Consider hospitalization",
            "Aggressive treatment may be necessary",
            "Regular bone marrow examination"
        ]
    }
    
    rec_data = [['CLINICAL RECOMMENDATIONS', '']]
    for rec in recommendations.get(detection.risk_level, []):
        rec_data.append(['', Paragraph(f"• {rec}", normal_style)])
    
    rec_table = Table(rec_data, colWidths=[2*inch, 4*inch])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(rec_table)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    # Save the PDF
    filename = f"report_{detection.patient.name}_{uuid.uuid4().hex[:8]}.pdf"
    filepath = os.path.join(settings.MEDIA_ROOT, 'reports', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        f.write(buffer.getvalue())
    
    detection.report_file.name = f"reports/{filename}"
    detection.report_generated = True
    detection.save()
    
    return redirect('patient_detail', pk=detection.patient.id)

@login_required
def download_report(request, detection_id):
    detection = get_object_or_404(DetectionResult, id=detection_id, patient__user=request.user)
    if not detection.report_generated:
        messages.error(request, 'Report not generated yet.')
        return redirect('patient_detail', pk=detection.patient.id)
    
    return FileResponse(open(detection.report_file.path, 'rb'), as_attachment=True)