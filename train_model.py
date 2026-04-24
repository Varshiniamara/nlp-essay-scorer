import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pickle
import textstat
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class EssayModelTrainer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        self.vectorizer = None
        self.dataset = None
        
    def load_dataset(self, file_path='training_set_rel3.csv'):
        """Load the ASAP dataset"""
        try:
            self.dataset = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with {len(self.dataset)} rows")
            print(f"Columns: {list(self.dataset.columns)}")
            return True
        except FileNotFoundError:
            print(f"Dataset file {file_path} not found. Creating sample dataset...")
            self.create_sample_dataset()
            return False
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def create_sample_dataset(self):
        """Create a sample dataset for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        # Sample essays with varying quality
        sample_essays = [
            "This is a good essay about education. Education is important for society. Students should study hard to achieve their goals.",
            "The impact of technology on modern education has been transformative. Digital tools have revolutionized how students learn and teachers instruct.",
            "Climate change represents one of the most significant challenges facing humanity today. The scientific evidence is overwhelming.",
            "In conclusion, while there are valid arguments on both sides, I believe that the benefits outweigh the drawbacks.",
            "The history of human civilization is marked by continuous innovation and adaptation to changing circumstances."
        ]
        
        # Generate essays by combining sample essays
        essays = []
        for _ in range(n_samples):
            essay = (np.random.choice(sample_essays) + " " + 
                    np.random.choice(sample_essays) + " " +
                    np.random.choice(sample_essays))
            essays.append(essay)
        
        data = {
            'essay_id': list(range(1, n_samples + 1)),
            'essay_set': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n_samples).tolist(),
            'essay': essays,
            'domain1_score': np.random.uniform(2, 12, n_samples).tolist()
        }
        
        self.dataset = pd.DataFrame(data)
        self.dataset.to_csv('training_set_rel3.csv', index=False)
        print("Sample dataset created and saved as training_set_rel3.csv")
    
    def preprocess_text(self, text):
        """Preprocess essay text"""
        if pd.isna(text):
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
    
    def extract_features(self, text):
        """Extract linguistic features for analysis"""
        if pd.isna(text):
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'char_count': 0,
                'vocab_richness': 0,
                'readability_score': 0,
                'grammar_errors': 0
            }
        
        text = str(text)
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Basic features
        word_count = len(words)
        sentence_count = len(sentences)
        char_count = len(text)
        
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Vocabulary richness
        unique_words = len(set(words))
        vocab_richness = unique_words / word_count if word_count > 0 else 0
        
        # Readability score
        try:
            readability_score = textstat.flesch_reading_ease(text)
        except:
            readability_score = 0
        
        # Simple grammar error estimation (basic approach)
        grammar_errors = self.estimate_grammar_errors(text)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'char_count': char_count,
            'vocab_richness': vocab_richness,
            'readability_score': readability_score,
            'grammar_errors': grammar_errors
        }
    
    def estimate_grammar_errors(self, text):
        """Simple grammar error estimation"""
        errors = 0
        
        # Check for common grammar patterns
        if re.search(r'\bi\s+[a-z]', text.lower()):  # Lowercase after "I"
            errors += 1
        if re.search(r'\s{2,}', text):  # Multiple spaces
            errors += 1
        if re.search(r'[.!?]{2,}', text):  # Multiple punctuation
            errors += 1
        if re.search(r'\b(a|an)\s+[aeiou]', text.lower()):  # Article misuse
            errors += 1
        
        return errors
    
    def prepare_data(self):
        """Prepare data for training"""
        print("Preprocessing data...")
        
        # Handle missing values
        self.dataset = self.dataset.dropna(subset=['essay', 'domain1_score'])
        
        # Preprocess essays
        self.dataset['processed_essay'] = self.dataset['essay'].apply(self.preprocess_text)
        
        # Extract features for analysis
        features_list = []
        for essay in self.dataset['essay']:
            features_list.append(self.extract_features(essay))
        
        features_df = pd.DataFrame(features_list)
        self.dataset = pd.concat([self.dataset, features_df], axis=1)
        
        # Prepare features and target
        X = self.dataset['processed_essay']
        y = self.dataset['domain1_score']
        
        return X, y
    
    def train_model(self):
        """Train the Random Forest model"""
        print("Preparing data for training...")
        X, y = self.prepare_data()
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Fit and transform training data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train Random Forest model
        print("Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_tfidf)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\nModel Evaluation Results:")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        
        # Save model and vectorizer
        self.save_model()
        
        # Generate graphs
        self.generate_graphs(X_test, y_test, y_pred)
        
        return {
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'model': self.model,
            'vectorizer': self.vectorizer
        }
    
    def save_model(self):
        """Save trained model and vectorizer"""
        try:
            with open('model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            with open('vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
            print("Model and vectorizer saved successfully!")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def generate_graphs(self, X_test, y_test, y_pred):
        """Generate visualization graphs"""
        # Create graphs directory
        os.makedirs('graphs', exist_ok=True)
        
        # Graph 1: Score Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.dataset['domain1_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Score Distribution in Dataset')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig('graphs/score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Graph 2: Word Count vs Score
        plt.figure(figsize=(10, 6))
        plt.scatter(self.dataset['word_count'], self.dataset['domain1_score'], 
                   alpha=0.6, color='coral')
        plt.title('Word Count vs Score')
        plt.xlabel('Word Count')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.savefig('graphs/wordcount_vs_score.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Graph 3: Grammar Errors vs Score
        plt.figure(figsize=(10, 6))
        plt.scatter(self.dataset['grammar_errors'], self.dataset['domain1_score'], 
                   alpha=0.6, color='lightgreen')
        plt.title('Grammar Errors vs Score')
        plt.xlabel('Grammar Errors')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.savefig('graphs/error_vs_score.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Graph 4: Feature Importance
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.vectorizer.get_feature_names_out()
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            plt.figure(figsize=(12, 8))
            plt.barh(importance_df['feature'], importance_df['importance'], color='purple')
            plt.title('Top 20 Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('graphs/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Graphs generated successfully in 'graphs' directory!")
    
    def load_trained_model(self):
        """Load pre-trained model and vectorizer"""
        try:
            with open('model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("Trained model and vectorizer loaded successfully!")
            return True
        except FileNotFoundError:
            print("Trained model not found. Please run train_model.py first.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    trainer = EssayModelTrainer()
    
    # Load dataset
    if not trainer.load_dataset():
        print("Using sample dataset for demonstration.")
    
    # Train model
    results = trainer.train_model()
    
    print("\nTraining completed successfully!")
    print(f"Model R² Score: {results['r2_score']:.4f}")
    print(f"Model MAE: {results['mae']:.4f}")
    print(f"Model RMSE: {results['rmse']:.4f}")

if __name__ == "__main__":
    main()
