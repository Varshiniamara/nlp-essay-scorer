import os
import sys
import traceback
from flask import Flask

app = Flask(__name__)

try:
    import nltk

    # Download required NLTK data to /tmp since AWS Lambda has read-only filesystem
    nltk_data_path = '/tmp/nltk_data'
    os.makedirs(nltk_data_path, exist_ok=True)
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)
    
    # Download quietly
    nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
    nltk.download('vader_lexicon', download_dir=nltk_data_path, quiet=True)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path, quiet=True)
    
    # punkt tab is necessary for modern strict textstat/nltk
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)

    # Add both root and nlp_project to sys.path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, 'nlp_project'))

    import nlp_project.app as main_app
    app = main_app.app

except Exception as e:
    err_trace = traceback.format_exc()
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def catch_all(path):
        return f"<pre>{err_trace}</pre>", 500
