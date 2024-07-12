# trigger_workflow.py
import sys
from dashboard.setup.celery_config import app, start_pipeline

if len(sys.argv) < 2:
    print("Usage: python trigger_workflow.py <csv_file_path>")
    sys.exit(1)

csv_file_path = sys.argv[1]

# Trigger the workflow
result = start_pipeline.delay(csv_file_path)

print(f"Task started with ID: {result.id}")
