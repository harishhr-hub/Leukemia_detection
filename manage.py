#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# Suppress TensorFlow oneDNN informational messages by default.
# This disables oneDNN optimizations which can change floating-point
# rounding behavior (useful for reproducible numerical results).
# Set to "0" to turn off oneDNN optimizations. Can be overridden by
# environment variables or user shell configuration.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'leukemia_detection.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()