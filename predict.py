import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import textstat
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='/tmp/nltk_data')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir='/tmp/nltk_data')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir='/tmp/nltk_data')

class EssayPredictor:
    def __init__(self, model_path=None, vectorizer_path=None):
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Resolve model paths relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base_dir, 'nlp_project', 'model.pkl')
            if not os.path.exists(model_path):
                model_path = os.path.join(base_dir, 'model.pkl')
        if vectorizer_path is None:
            vectorizer_path = os.path.join(base_dir, 'nlp_project', 'vectorizer.pkl')
            if not os.path.exists(vectorizer_path):
                vectorizer_path = os.path.join(base_dir, 'vectorizer.pkl')
        
        # Load trained model and vectorizer
        self.load_model(model_path, vectorizer_path)
    
    def load_model(self, model_path, vectorizer_path):
        """Load trained model and vectorizer"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("Trained model and vectorizer loaded successfully!")
            return True
        except FileNotFoundError:
            print("Model files not found. Please run train_model.py first.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_text(self, text):
        """Preprocess essay text for prediction"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_detailed_features(self, text):
        """Extract comprehensive linguistic features"""
        if not text or pd.isna(text):
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'char_count': 0,
                'vocab_richness': 0,
                'readability_score': 0,
                'grammar_errors': 0,
                'avg_sentence_length': 0,
                'complex_words': 0,
                'syllable_count': 0
            }
        
        text = str(text)
        words = text.split()
        sentences = sent_tokenize(text)
        
        # Basic features
        word_count = len(words)
        sentence_count = len(sentences)
        char_count = len(text)
        
        # Word-level features
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Vocabulary richness
        unique_words = len(set(words))
        vocab_richness = unique_words / word_count if word_count > 0 else 0
        
        # Readability and complexity
        try:
            readability_score = textstat.flesch_reading_ease(text)
            complex_words = textstat.difficult_words(text)
            syllable_count = textstat.syllable_count(text)
        except:
            readability_score = 0
            complex_words = 0
            syllable_count = 0
        
        # Grammar error estimation
        grammar_errors = self.estimate_grammar_errors(text)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'char_count': char_count,
            'vocab_richness': vocab_richness,
            'readability_score': readability_score,
            'grammar_errors': grammar_errors,
            'avg_sentence_length': avg_sentence_length,
            'complex_words': complex_words,
            'syllable_count': syllable_count
        }
    
    def estimate_grammar_errors(self, text):
        """Estimate grammar errors in text"""
        if not text:
            return 0
        
        errors = 0
        text_lower = text.lower()
        
        # Check for common grammar patterns
        patterns = [
            (r'\bi\s+[a-z]', 'Lowercase after "I"'),
            (r'\s{2,}', 'Multiple spaces'),
            (r'[.!?]{2,}', 'Multiple punctuation'),
            (r'\b(a|an)\s+[aeiou]', 'Article misuse'),
            (r'\bthe\s+the\b', 'Repeated "the"'),
            (r'\s+[,.]', 'Space before punctuation'),
            (r'[,.]\s*[a-z]', 'Lowercase after punctuation'),
            (r'\b(is|are|was|were)\s+\w+ing\b', 'Incorrect verb tense'),
            (r'\bvery\s+very\b', 'Repeated "very"'),
            (r'\bbecause\s+because\b', 'Repeated "because"')
        ]
        
        for pattern, description in patterns:
            matches = re.findall(pattern, text_lower)
            errors += len(matches)
        
        # Additional checks
        if text.count('.') != text.count('. '):  # Missing space after period
            errors += 1
        
        return errors
    
    def predict_score(self, text):
        """Predict essay score with detailed analysis"""
        if not text or len(text.strip()) < 10:
            return {
                'score': 1.0,
                'error': 'Essay too short for meaningful evaluation',
                'features': None,
                'feedback': None
            }
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Extract features
            features = self.extract_detailed_features(text)
            
            # Predict score using trained model
            if self.model and self.vectorizer:
                text_vector = self.vectorizer.transform([processed_text])
                predicted_score = self.model.predict(text_vector)[0]
                predicted_score = max(1, min(12, float(predicted_score)))  # Clamp to valid range
            else:
                # Fallback scoring based on features
                predicted_score = self.calculate_fallback_score(features)
            
            # Generate detailed feedback
            feedback = self.generate_comprehensive_feedback(features, predicted_score)
            
            return {
                'score': round(predicted_score, 1),
                'features': features,
                'feedback': feedback,
                'error': None
            }
            
        except Exception as e:
            return {
                'score': 5.0,
                'error': f'Error during prediction: {str(e)}',
                'features': None,
                'feedback': 'An error occurred during scoring. Please try again.'
            }
    
    def calculate_fallback_score(self, features):
        """Calculate fallback score based on linguistic features"""
        score = 0
        
        # Word count contribution (0-3 points)
        if features['word_count'] >= 300:
            score += 3
        elif features['word_count'] >= 200:
            score += 2
        elif features['word_count'] >= 100:
            score += 1
        
        # Sentence count contribution (0-2 points)
        if features['sentence_count'] >= 15:
            score += 2
        elif features['sentence_count'] >= 10:
            score += 1
        
        # Vocabulary richness contribution (0-2 points)
        if features['vocab_richness'] >= 0.6:
            score += 2
        elif features['vocab_richness'] >= 0.4:
            score += 1
        
        # Readability contribution (0-2 points)
        if features['readability_score'] >= 60:
            score += 2
        elif features['readability_score'] >= 40:
            score += 1
        
        # Grammar quality contribution (0-3 points)
        if features['grammar_errors'] <= 2:
            score += 3
        elif features['grammar_errors'] <= 5:
            score += 2
        elif features['grammar_errors'] <= 10:
            score += 1
        
        # Ensure score is within bounds
        return max(1, min(12, score))
    
    def generate_comprehensive_feedback(self, features, score):
        """Generate detailed feedback based on analysis"""
        feedback = {
            'overall_assessment': '',
            'strengths': [],
            'weaknesses': [],
            'suggestions': [],
            'detailed_metrics': {}
        }
        
        # Overall assessment
        if score >= 10:
            feedback['overall_assessment'] = "Excellent essay! Outstanding quality across all dimensions."
        elif score >= 8:
            feedback['overall_assessment'] = "Very good essay with strong content and structure."
        elif score >= 6:
            feedback['overall_assessment'] = "Good essay meeting most requirements effectively."
        elif score >= 4:
            feedback['overall_assessment'] = "Acceptable essay that meets basic requirements."
        else:
            feedback['overall_assessment'] = "Essay needs significant improvement."
        
        # Analyze strengths and weaknesses
        if features['word_count'] >= 250:
            feedback['strengths'].append("Good essay length with comprehensive content")
        elif features['word_count'] < 100:
            feedback['weaknesses'].append("Essay is too short - needs more development")
        
        if features['vocab_richness'] >= 0.5:
            feedback['strengths'].append("Excellent vocabulary diversity")
        elif features['vocab_richness'] < 0.3:
            feedback['weaknesses'].append("Limited vocabulary - use more varied words")
        
        if features['grammar_errors'] <= 3:
            feedback['strengths'].append("Good grammar with minimal errors")
        elif features['grammar_errors'] > 8:
            feedback['weaknesses'].append(f"Multiple grammar errors detected ({features['grammar_errors']})")
        
        if features['readability_score'] >= 50:
            feedback['strengths'].append("Good readability and flow")
        elif features['readability_score'] < 30:
            feedback['weaknesses'].append("Difficult to read - improve sentence structure")
        
        # Generate suggestions
        if features['word_count'] < 200:
            feedback['suggestions'].append("Expand essay with more examples and details")
        
        if features['vocab_richness'] < 0.4:
            feedback['suggestions'].append("Use more diverse and sophisticated vocabulary")
        
        if features['grammar_errors'] > 5:
            feedback['suggestions'].append("Review and correct grammar errors")
        
        if features['readability_score'] < 40:
            feedback['suggestions'].append("Improve sentence structure for better readability")
        
        if features['avg_sentence_length'] > 25:
            feedback['suggestions'].append("Break down long sentences for clarity")
        
        # Detailed metrics
        feedback['detailed_metrics'] = {
            'word_count': features['word_count'],
            'sentence_count': features['sentence_count'],
            'vocab_richness': round(features['vocab_richness'], 3),
            'readability_score': round(features['readability_score'], 1),
            'grammar_errors': features['grammar_errors'],
            'avg_word_length': round(features['avg_word_length'], 1),
            'complex_words': features['complex_words']
        }
        
        return feedback

# Global predictor instance
predictor = EssayPredictor()

def predict_score(text):
    """Global function to predict essay score"""
    return predictor.predict_score(text)

# Import pandas for pd.isna check
import pandas as pd
