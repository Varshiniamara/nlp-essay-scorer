import os
import sys

# Ensure both the root and nlp_project dirs are on sys.path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, 'nlp_project'))

from nlp_project.app import app

handler = app
