# Acute Lymphoblastic Leukemia Detection System

This is a web-based application for detecting Acute Lymphoblastic Leukemia (ALL) using deep learning. The system provides risk assessment and generates detailed reports for healthcare professionals.

## Features

- User authentication and authorization
- Patient management system
- Blood smear image analysis using deep learning
- Risk level assessment
- PDF report generation
- Interactive and responsive UI

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd leukemia_detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up the database:
```bash
python manage.py makemigrations
python manage.py migrate
```

5. Create a superuser:
```bash
python manage.py createsuperuser
```

6. Create a `model` directory and add your trained model:
```bash
mkdir model
# Add your trained ALL detection model as 'all_detection_model.h5'
```

7. Run the development server:
```bash
python manage.py runserver
```

The application will be available at `http://localhost:8000`

## Project Structure

```
leukemia_detection/
├── detection_app/          # Main application
│   ├── models.py          # Database models
│   ├── views.py           # View functions
│   ├── forms.py           # Form definitions
│   ├── urls.py            # URL patterns
│   └── admin.py           # Admin interface
├── static/                # Static files
│   └── css/              # CSS files
├── templates/             # HTML templates
│   └── detection_app/    # App-specific templates
├── media/                # User-uploaded files
├── model/                # ML model directory
└── manage.py            # Django management script
```

## Usage

1. Register a new account or login
2. Add patient information
3. Upload blood smear images for analysis
4. View detection results and risk assessment
5. Generate and download detailed reports

## Note

Make sure to add your trained model file `all_detection_model.h5` in the `model` directory before running the application."# leukemia" 
