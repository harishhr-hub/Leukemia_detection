from django.contrib import admin
from .models import Patient, DetectionResult, PatientFollowUp, SurvivalAnalysis

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('name', 'age', 'gender', 'created_at')
    search_fields = ('name', 'medical_history')
    list_filter = ('gender', 'created_at')
    fieldsets = (
        ('Personal Information', {
            'fields': ('user', 'name', 'age', 'gender', 'medical_history')
        }),
    )

@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = ('patient', 'detection_result', 'risk_level', 'date_detected')
    list_filter = ('risk_level', 'date_detected', 'report_generated')
    search_fields = ('patient__name', 'detection_result')

@admin.register(PatientFollowUp)
class PatientFollowUpAdmin(admin.ModelAdmin):
    list_display = ('patient', 'visit_date', 'event_status', 'survival_days')
    list_filter = ('event_status', 'visit_date')
    search_fields = ('patient__name',)

@admin.register(SurvivalAnalysis)
class SurvivalAnalysisAdmin(admin.ModelAdmin):
    list_display = ('patient', 'risk_score', 'risk_group', 'predicted_survival_months')
    list_filter = ('risk_group', 'created_at')
    search_fields = ('patient__name',)