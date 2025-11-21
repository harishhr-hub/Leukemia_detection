from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("detection_app", "0002_detectionresult_tta_probs"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="detectionresult",
            name="confidence_score",
        ),
    ]
