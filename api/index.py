import os
import sys

# Add root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# Now import the app from nlp_project/app.py
from nlp_project.app import app

# Vercel needs the app as 'app' or 'handler'
# If app is already 'app', we are good.
handler = app
