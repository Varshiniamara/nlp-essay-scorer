from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import json

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from predict import predict_score
from enhanced_predict import analyze_essay_with_mistakes
from advanced_nlp_analyzer import analyze_essay_comprehensive_nlp
from ml_scoring_engine import analyze_essay_with_ml

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_in_production'

# Database configuration
if os.environ.get('VERCEL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/essay_scoring.db'
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///essay_scoring.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with essays
    essays = db.relationship('Essay', backref='author', lazy=True, cascade='all, delete-orphan')

class Essay(db.Model):
    __tablename__ = 'essays'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    essay_text = db.Column(db.Text, nullable=False)
    submission_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with results
    results = db.relationship('Result', backref='essay', lazy=True, cascade='all, delete-orphan')

class Result(db.Model):
    __tablename__ = 'results'
    id = db.Column(db.Integer, primary_key=True)
    essay_id = db.Column(db.Integer, db.ForeignKey('essays.id'), nullable=False)
    score = db.Column(db.Float, nullable=False)
    feedback = db.Column(db.Text, nullable=False)
    grammar_errors = db.Column(db.Integer, default=0)
    word_count = db.Column(db.Integer, default=0)
    readability_score = db.Column(db.Float, default=0.0)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

# Create database tables
with app.app_context():
    db.create_all()

# Helper Functions
def is_logged_in():
    return 'user_id' in session

def get_current_user():
    if is_logged_in():
        return User.query.get(session['user_id'])
    return None

def get_dashboard_data():
    """Get data for dashboard graphs"""
    if not is_logged_in():
        return None
    
    current_user = get_current_user()
    essays = Essay.query.filter_by(user_id=current_user.id).join(Result).all()
    
    if not essays:
        return None
    
    # Prepare data for graphs
    scores = [result.results[0].score for result in essays if result.results]
    word_counts = [result.results[0].word_count for result in essays if result.results]
    grammar_errors = [result.results[0].grammar_errors for result in essays if result.results]
    
    return {
        'scores': scores,
        'word_counts': word_counts,
        'grammar_errors': grammar_errors,
        'essay_count': len(essays),
        'avg_score': sum(scores) / len(scores) if scores else 0,
        'total_words': sum(word_counts) if word_counts else 0
    }

# Routes
@app.route('/')
def index():
    if is_logged_in():
        return render_template('index.html', user=get_current_user())
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not name or not email or not password:
            flash('All fields are required', 'error')
        elif password != confirm_password:
            flash('Passwords do not match', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
        else:
            # Check if user already exists
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash('Email already registered', 'error')
            else:
                # Create new user
                hashed_password = generate_password_hash(password)
                new_user = User(name=name, email=email, password=hashed_password)
                
                db.session.add(new_user)
                db.session.commit()
                
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if not is_logged_in():
        flash('Please login to submit an essay', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        essay_text = request.form.get('essay_text')
        
        if not essay_text or len(essay_text.strip()) < 10:
            flash('Essay must be at least 10 characters long', 'error')
            return render_template('index.html', user=get_current_user())
        
        try:
            # Get current user
            current_user = get_current_user()
            
            if not current_user:
                flash('User session expired. Please login again.', 'error')
                return redirect(url_for('login'))
            
            # Predict score with comprehensive NLP analysis
            # Get reference essays for semantic similarity (from database)
            reference_essays = []
            reference_scores = []
            
            # Try to get recent essays for semantic comparison
            try:
                recent_essays = Essay.query.filter(Essay.user_id != current_user.id)\
                                     .join(Result)\
                                     .order_by(Essay.submission_date.desc())\
                                     .limit(5).all()
                
                for essay in recent_essays:
                    if essay.results:
                        reference_essays.append(essay.essay_text)
                        reference_scores.append(essay.results[0].score)
            except:
                reference_essays = []
                reference_scores = []
            
            # Perform ML-based essay analysis with NLP components
            ml_analysis = analyze_essay_with_ml(essay_text, reference_essays)
            
            # Debug: Log ML analysis result
            print(f"DEBUG: ML-Based NLP Analysis:")
            print(f"  - Predicted Score: {ml_analysis.get('score')}")
            print(f"  - Confidence: {ml_analysis.get('confidence', 0):.3f}")
            print(f"  - Tokenization: {ml_analysis.get('features', {}).get('tokenization', {}).get('token_count', 0)} tokens")
            print(f"  - Lexical Diversity: {ml_analysis.get('features', {}).get('lexical_analysis', {}).get('lexical_diversity', 0):.3f}")
            print(f"  - Readability: {ml_analysis.get('features', {}).get('readability', {}).get('flesch_reading_ease', 0):.1f}")
            print(f"  - Vector Space: {ml_analysis.get('features', {}).get('vector_space', {}).get('vocabulary_size', 0)} dimensions")
            
            if ml_analysis.get('error'):
                flash(ml_analysis.get('error'), 'error')
                return render_template('index.html', user=get_current_user())
            
            # Save essay to database
            new_essay = Essay(
                user_id=current_user.id,
                essay_text=essay_text
            )
            db.session.add(new_essay)
            db.session.commit()
            
            # Debug: Check if essay was created successfully
            if not new_essay or not new_essay.id:
                flash('Failed to create essay record', 'error')
                return render_template('index.html', user=get_current_user())
            
            # Extract ML analysis components
            features = ml_analysis.get('features', {})
            recommendations = ml_analysis.get('recommendations', {})
            lexical_analysis = features.get('lexical_analysis', {})
            readability = features.get('readability', {})
            vector_space = features.get('vector_space', {})
            
            # Debug: Log ML analysis details
            print(f"DEBUG: ML Analysis Components:")
            print(f"  - Score: {ml_analysis.get('score')}")
            print(f"  - Confidence: {ml_analysis.get('confidence', 0):.3f}")
            print(f"  - Lexical Diversity: {lexical_analysis.get('lexical_diversity', 0):.3f}")
            print(f"  - Readability: {readability.get('flesch_reading_ease', 0):.1f}")
            print(f"  - Vector Space: {vector_space.get('vocabulary_size', 0)} dimensions")
            
            # Save ML result to database
            try:
                new_result = Result(
                    essay_id=new_essay.id,
                    score=ml_analysis.get('score', 0.0),
                    feedback=json.dumps(ml_analysis),
                    grammar_errors=0,  # Will be calculated from ML analysis
                    word_count=lexical_analysis.get('total_words', 0),
                    readability_score=readability.get('flesch_reading_ease', 0.0)
                )
                db.session.add(new_result)
                db.session.commit()
                print(f"DEBUG: ML result saved with essay_id: {new_essay.id}")
                print(f"DEBUG: Score: {ml_analysis.get('score')}")
                print(f"DEBUG: Confidence: {ml_analysis.get('confidence', 0):.3f}")
            except Exception as db_error:
                print(f"DEBUG: Database error: {db_error}")
                db.session.rollback()
                flash(f'Database error: {str(db_error)}', 'error')
                return render_template('index.html', user=get_current_user())
            
            # Redirect to result page
            return redirect(url_for('result', essay_id=new_essay.id))
            
        except Exception as e:
            flash(f'An error occurred while processing your essay: {str(e)}', 'error')
            return render_template('index.html', user=get_current_user())
    
    return render_template('index.html', user=get_current_user())

@app.route('/result/<int:essay_id>')
def result(essay_id):
    if not is_logged_in():
        flash('Please login to view results', 'error')
        return redirect(url_for('login'))
    
    # Get essay and result
    essay = Essay.query.get_or_404(essay_id)
    result = Result.query.filter_by(essay_id=essay_id).first()
    
    # Check if the essay belongs to current user
    if essay.user_id != session['user_id']:
        flash('Access denied', 'error')
        return redirect(url_for('index'))
    
    if not result:
        flash('Result not found', 'error')
        return redirect(url_for('index'))
    
    # Parse ML analysis feedback
    try:
        if result.feedback and result.feedback.startswith('['):
            feedback = eval(result.feedback)
        else:
            # Try to parse as JSON if it's not old format
            import json
            feedback = json.loads(result.feedback) if result.feedback else {}
    except:
        feedback = {}
    
    # Debug: Check feedback structure
    print(f"DEBUG: Feedback structure: {type(feedback)}")
    if isinstance(feedback, dict):
        print(f"DEBUG: Feedback keys: {list(feedback.keys())}")
    
    # Fix ML analysis structure
    ml_analysis = {
        'score': result.score,
        'confidence': 0.0,  # Replace with actual confidence value
        'features': {
            'lexical_analysis': {
                'lexical_diversity': 0.0,  # Replace with actual lexical diversity value
                'total_words': result.word_count
            },
            'readability': {
                'flesch_reading_ease': result.readability_score
            },
            'vector_space': {
                'vocabulary_size': 0  # Replace with actual vocabulary size value
            }
        },
        'recommendations': feedback
    }
    
    return render_template('enhanced_result_template.html', essay=essay, result=result, feedback=feedback, user=get_current_user())

@app.route('/history')
def history():
    if not is_logged_in():
        flash('Please login to view your history', 'error')
        return redirect(url_for('login'))
    
    current_user = get_current_user()
    
    # Get all essays with results for current user
    essays = Essay.query.filter_by(user_id=current_user.id)\
                       .join(Result)\
                       .order_by(Essay.submission_date.desc())\
                       .all()
    
    return render_template('history.html', essays=essays, user=get_current_user())

@app.route('/dashboard')
def dashboard():
    if not is_logged_in():
        flash('Please login to view dashboard', 'error')
        return redirect(url_for('login'))
    
    dashboard_data = get_dashboard_data()
    
    if not dashboard_data:
        return render_template('dashboard.html', 
                           no_data=True, 
                           user=get_current_user())
    
    return render_template('dashboard.html', 
                       data=dashboard_data, 
                       user=get_current_user())

@app.route('/delete_essay/<int:essay_id>', methods=['POST'])
def delete_essay(essay_id):
    if not is_logged_in():
        flash('Please login to delete essays', 'error')
        return redirect(url_for('history'))
    
    essay = Essay.query.get_or_404(essay_id)
    
    # Check if the essay belongs to current user
    if essay.user_id != session['user_id']:
        flash('Access denied', 'error')
        return redirect(url_for('history'))
    
    try:
        # Delete essay (this will also delete associated results due to cascade)
        db.session.delete(essay)
        db.session.commit()
        flash('Essay deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting essay: {str(e)}', 'error')
    
    return redirect(url_for('history'))

@app.route('/api/essay_stats')
def api_essay_stats():
    """API endpoint for essay statistics"""
    if not is_logged_in():
        return jsonify({'error': 'Not logged in'}), 401
    
    dashboard_data = get_dashboard_data()
    
    if not dashboard_data:
        return jsonify({'error': 'No data available'}), 404
    
    return jsonify(dashboard_data)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)
