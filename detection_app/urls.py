from django.urls import path
from . import views
from django.contrib.auth.views import LoginView, LogoutView

urlpatterns = [
    path('', views.home, name='home'),
    path('signup/', views.signup, name='signup'),
    path('login/', LoginView.as_view(template_name='detection_app/login.html'), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('patient/add/', views.add_patient, name='add_patient'),
    path('patient/<int:pk>/', views.patient_detail, name='patient_detail'),
    path('detect/<int:patient_id>/', views.detect_leukemia, name='detect_leukemia'),
    path('generate-report/<int:detection_id>/', views.generate_report, name='generate_report'),
    path('download-report/<int:detection_id>/', views.download_report, name='download_report'),
]