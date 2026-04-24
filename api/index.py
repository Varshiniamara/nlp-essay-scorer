import os
import sys
import nltk

# Set NLTK data path to bundled folder in repo
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nltk_data_path = os.path.join(root, 'nltk_data')
if nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_path)

# Add both root and nlp_project to sys.path
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, 'nlp_project'))

from nlp_project.app import app

handler = app
