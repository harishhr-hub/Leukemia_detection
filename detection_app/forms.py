from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Patient, DetectionResult

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')

class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ['name', 'age', 'gender', 'medical_history']
        widgets = {
            'medical_history': forms.Textarea(attrs={'rows': 3}),
        }

class DetectionForm(forms.ModelForm):
    class Meta:
        model = DetectionResult
        fields = ['image']