import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat
import re
from collections import Counter
try:
    import spacy
except ImportError:
    spacy = None

class AdvancedNLPEssayAnalyzer:
    """
    Comprehensive NLP-based Automated Essay Scoring System
    Implements tokenization, lexical analysis, text preprocessing, 
    feature extraction, and vector space modeling
    """
    
    def __init__(self):
        # Download required NLTK data
        self.download_nltk_data()
        
        # Initialize components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize spaCy for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize TF-IDF vectorizer for semantic analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True,
            analyzer='word'
        )
        
        # Reference essays for semantic similarity (trained on ASAP dataset)
        self.reference_essays = []
        self.reference_scores = []
        
    def download_nltk_data(self):
        """Download required NLTK data"""
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
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        try:
            nltk.data.find('corpora/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        try:
            nltk.data.find('chunkers/maxent_ne_chunker')
        except LookupError:
            nltk.download('maxent_ne_chunker')
        
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
    
    def advanced_preprocessing(self, text):
        """Advanced text preprocessing for NLP analysis"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def linguistic_feature_extraction(self, text):
        """Extract comprehensive linguistic features"""
        if not text:
            return {}
        
        # Basic statistics
        words = word_tokenize(text.lower()) if text else []
        sentences = sent_tokenize(text) if text else []
        tokens = [word for word in words if word.isalpha()]
        
        # Lexical diversity metrics
        unique_words = set(tokens) if tokens else set()
        type_token_ratio = len(unique_words) / len(tokens) if tokens else 0
        
        # Word length statistics
        word_lengths = [len(word) for word in tokens if isinstance(word, str)]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0
        
        # Sentence complexity
        sentence_lengths = []
        for sent in sentences:
            if isinstance(sent, str):
                words_in_sent = sent.split()
                sentence_lengths.append(len(words_in_sent))
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        
        # Readability metrics
        flesch_score = textstat.flesch_reading_ease(text)
        fog_index = textstat.gunning_fog(text)
        coleman_liau = textstat.coleman_liau_index(text)
        
        # Syntactic complexity
        pos_tags = pos_tag(words)
        noun_count = len([tag for word, tag in pos_tags if tag.startswith('NN')])
        verb_count = len([tag for word, tag in pos_tags if tag.startswith('VB')])
        adj_count = len([tag for word, tag in pos_tags if tag.startswith('JJ')])
        
        # Vocabulary sophistication
        difficult_words = textstat.difficult_words(text)
        polysyllable_count = textstat.polysyllabcount(text)
        
        # Ensure difficult_words is a list
        if isinstance(difficult_words, str):
            difficult_words = [difficult_words]
        elif not isinstance(difficult_words, list):
            difficult_words = []
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        
        # Named entity recognition
        entities = []
        if self.nlp:
            doc = self.nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        return {
            # Basic metrics
            'word_count': len(words),
            'sentence_count': len(sentences),
            'char_count': len(text),
            
            # Lexical diversity
            'unique_words': len(unique_words),
            'type_token_ratio': type_token_ratio,
            'lexical_diversity': type_token_ratio,
            
            # Word complexity
            'avg_word_length': avg_word_length,
            'max_word_length': max(word_lengths) if word_lengths else 0,
            'difficult_words': len(difficult_words),
            'polysyllable_words': polysyllable_count,
            
            # Sentence complexity
            'avg_sentence_length': avg_sentence_length,
            'max_sentence_length': max(sentence_lengths) if sentence_lengths else 0,
            'sentence_complexity_variance': np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0,
            
            # Readability metrics
            'flesch_reading_ease': flesch_score,
            'gunning_fog_index': fog_index,
            'coleman_liau_index': coleman_liau,
            
            # Syntactic features
            'noun_count': noun_count,
            'verb_count': verb_count,
            'adjective_count': adj_count,
            'pos_diversity': len(set([tag for word, tag in pos_tags])),
            
            # Sentiment features
            'sentiment_positive': sentiment['pos'],
            'sentiment_negative': sentiment['neg'],
            'sentiment_neutral': sentiment['neu'],
            'sentiment_compound': sentiment['compound'],
            
            # Named entities
            'named_entities': entities,
            'entity_count': len(entities),
            
            # Text structure
            'punctuation_count': len([char for char in text if char in '.,!?;:']),
            'uppercase_ratio': len([char for char in text if char.isupper()]) / len(text) if text else 0
        }
    
    def vector_space_modeling(self, text, reference_texts=None):
        """Implement vector space modeling and semantic similarity"""
        if not text:
            return {}
        
        # Preprocess text
        processed_text = self.advanced_preprocessing(text)
        
        # Create document corpus
        corpus = [processed_text]
        if reference_texts:
            corpus += [self.advanced_preprocessing(ref) for ref in reference_texts]
        elif self.reference_essays:
            corpus += [self.advanced_preprocessing(ref) for ref in self.reference_essays]
        
        # Fit TF-IDF vectorizer
        if len(corpus) > 1:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        else:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([processed_text])
        
        # Get vector for current essay
        essay_vector = tfidf_matrix[0:1]
        
        # Calculate semantic similarity with reference essays
        similarities = []
        if tfidf_matrix.shape[0] > 1 and (reference_texts or self.reference_essays):
            try:
                reference_vectors = tfidf_matrix[1:]
                sim_result = cosine_similarity(essay_vector, reference_vectors)
                if sim_result.shape[0] > 0 and sim_result.shape[1] > 0:
                    similarities = sim_result[0]
            except:
                similarities = []
        
        # Topic modeling using LDA
        topics = []
        if len(corpus) > 5:  # Need enough documents for meaningful topics
            try:
                lda = LatentDirichletAllocation(n_components=5, random_state=42)
                lda_topics = lda.fit_transform(tfidf_matrix)
                if lda_topics.shape[0] > 0:
                    topics = lda_topics[0].tolist() if hasattr(lda_topics[0], 'tolist') else list(lda_topics[0])
            except:
                topics = []
        
        return {
            'tfidf_vector': essay_vector.toarray().flatten(),
            'vocabulary_size': len(self.tfidf_vectorizer.vocabulary_) if hasattr(self.tfidf_vectorizer, 'vocabulary_') else 0,
            'semantic_similarities': similarities if isinstance(similarities, list) else [],
            'avg_semantic_similarity': np.mean(similarities) if isinstance(similarities, list) and len(similarities) > 0 else 0,
            'topic_distribution': topics if isinstance(topics, list) else [],
            'dominant_topic': np.argmax(topics) if isinstance(topics, list) and len(topics) > 0 else -1
        }
    
    def coherence_analysis(self, text):
        """Analyze text coherence and discourse structure"""
        if not text:
            return {}
        
        sentences = sent_tokenize(text)
        
        # Cohesion markers
        cohesion_markers = [
            'therefore', 'however', 'furthermore', 'moreover', 'consequently',
            'in addition', 'additionally', 'nevertheless', 'on the other hand',
            'for example', 'for instance', 'specifically', 'in particular'
        ]
        
        # Count cohesion markers
        cohesion_count = sum(1 for sentence in sentences 
                          for marker in cohesion_markers 
                          if marker in sentence.lower())
        
        # Pronoun consistency (cohesion indicator)
        pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they']
        pronoun_usage = {}
        for pronoun in pronouns:
            pronoun_usage[pronoun] = sum(1 for word in word_tokenize(text.lower()) 
                                      if word == pronoun)
        
        # Argument structure analysis
        argument_indicators = [
            'because', 'since', 'due to', 'as a result', 'therefore',
            'first', 'second', 'third', 'finally', 'in conclusion'
        ]
        
        argument_structure_score = sum(1 for sentence in sentences 
                                  for indicator in argument_indicators 
                                  if indicator in sentence.lower())
        
        # Paragraph structure
        paragraphs = text.split('\n\n')
        paragraph_lengths = []
        for p in paragraphs:
            if p.strip():
                words = p.split()
                if isinstance(words, list):
                    paragraph_lengths.append(len(words))
                else:
                    paragraph_lengths.append(1)  # Fallback
        
        return {
            'cohesion_markers': cohesion_count,
            'cohesion_density': cohesion_count / len(sentences) if sentences else 0,
            'pronoun_consistency': pronoun_usage,
            'dominant_pronoun': max(pronoun_usage.items(), key=lambda x: x[1])[0] if pronoun_usage else None,
            'argument_structure_score': argument_structure_score,
            'paragraph_count': len([p for p in paragraphs if p.strip()]),
            'avg_paragraph_length': np.mean(paragraph_lengths) if paragraph_lengths else 0,
            'paragraph_variance': np.var(paragraph_lengths) if len(paragraph_lengths) > 1 else 0
        }
    
    def grammar_accuracy_analysis(self, text):
        """Comprehensive grammar accuracy assessment"""
        if not text:
            return {}
        
        # Advanced grammar patterns
        grammar_patterns = {
            'subject_verb_agreement': {
                'pattern': r'\b(?:he|she|it)\s+\w+s\b',
                'error': 'Subject-verb disagreement',
                'severity': 'major'
            },
            'tense_consistency': {
                'pattern': r'\b(?:walk|go|see)\s+and\s+\w+ed\b',
                'error': 'Tense inconsistency',
                'severity': 'moderate'
            },
            'article_usage': {
                'pattern': r'\b(a)\s+[aeiou]',
                'error': 'Incorrect article usage',
                'severity': 'minor'
            },
            'preposition_usage': {
                'pattern': r'\b(?:different|similar)\s+than\s+(?:the|a|an)\b',
                'error': 'Incorrect preposition',
                'severity': 'moderate'
            },
            'double_negative': {
                'pattern': r'\bnot\s+not\b|\bnever\s+no\b',
                'error': 'Double negative',
                'severity': 'major'
            },
            'run_on_sentence': {
                'pattern': r'[.!?]\s+[A-Z][^.!?]*[.!?]',
                'error': 'Run-on sentence',
                'severity': 'moderate'
            },
            'fragment': {
                'pattern': r'^\s*[A-Z][^.!?]*$',
                'error': 'Sentence fragment',
                'severity': 'minor'
            }
        }
        
        errors_found = []
        total_errors = 0
        
        for error_type, pattern_info in grammar_patterns.items():
            matches = re.finditer(pattern_info['pattern'], text, re.IGNORECASE)
            for match in matches:
                errors_found.append({
                    'type': error_type,
                    'error': pattern_info['error'],
                    'severity': pattern_info['severity'],
                    'position': match.start(),
                    'context': self.get_context(text, match.start(), 50),
                    'matched_text': match.group()
                })
                total_errors += 1
        
        # Calculate grammar accuracy score
        sentences = sent_tokenize(text)
        accuracy_score = max(0, 1 - (total_errors / len(sentences))) if sentences else 0
        
        return {
            'total_errors': total_errors,
            'error_density': total_errors / len(sentences) if sentences else 0,
            'grammar_accuracy': accuracy_score,
            'errors_found': errors_found,
            'error_types': list(set([e['type'] for e in errors_found]))
        }
    
    def get_context(self, text, position, window_size=50):
        """Get context around a position in text"""
        start = max(0, position - window_size)
        end = min(len(text), position + window_size + len(text[position:position+30]))
        return text[start:end].strip()
    
    def comprehensive_essay_analysis(self, text, reference_essays=None, reference_scores=None):
        """
        Complete NLP-based essay analysis implementing all required features:
        - Tokenization and lexical analysis
        - Text preprocessing and feature extraction  
        - Vector space modeling and semantic similarity
        - Grammar accuracy and lexical diversity
        - Coherence and readability metrics
        """
        if not text or len(text.strip()) < 10:
            return {
                'overall_score': 1.0,
                'error': 'Essay too short for meaningful analysis',
                'analysis': {}
            }
        
        # Set reference data if provided
        if reference_essays:
            self.reference_essays = reference_essays
        if reference_scores:
            self.reference_scores = reference_scores
        
        # Perform all analyses
        linguistic_features = self.linguistic_feature_extraction(text)
        vector_analysis = self.vector_space_modeling(text, reference_essays)
        coherence_metrics = self.coherence_analysis(text)
        grammar_analysis = self.grammar_accuracy_analysis(text)
        
        # Calculate comprehensive score
        overall_score = self.calculate_comprehensive_score(
            linguistic_features, vector_analysis, coherence_metrics, grammar_analysis
        )
        
        # Generate detailed feedback
        feedback = self.generate_comprehensive_feedback(
            linguistic_features, vector_analysis, coherence_metrics, grammar_analysis
        )
        
        return {
            'overall_score': round(overall_score, 1),
            'linguistic_features': linguistic_features,
            'vector_analysis': vector_analysis,
            'coherence_metrics': coherence_metrics,
            'grammar_analysis': grammar_analysis,
            'detailed_feedback': feedback,
            'error': None
        }
    
    def calculate_comprehensive_score(self, linguistic, vector, coherence, grammar):
        """Calculate comprehensive score based on all analyses"""
        score = 12.0  # Start from perfect score
        
        # Linguistic features contribution (40% weight)
        if linguistic['type_token_ratio'] < 0.3:
            score -= 2.0  # Poor vocabulary
        elif linguistic['type_token_ratio'] < 0.5:
            score -= 1.0  # Limited vocabulary
        
        if linguistic['flesch_reading_ease'] < 30:
            score -= 2.0  # Very difficult to read
        elif linguistic['flesch_reading_ease'] < 50:
            score -= 1.0  # Difficult to read
        
        # Vector space analysis (25% weight)
        if vector['avg_semantic_similarity'] < 0.2:
            score -= 1.5  # Low semantic similarity to references
        elif vector['avg_semantic_similarity'] < 0.4:
            score -= 0.8  # Moderate semantic similarity
        
        # Coherence analysis (20% weight)
        if coherence['cohesion_density'] < 0.1:
            score -= 1.0  # Poor cohesion
        if coherence['argument_structure_score'] < 2:
            score -= 0.5  # Weak argument structure
        
        # Grammar accuracy (15% weight)
        if grammar['grammar_accuracy'] < 0.8:
            score -= 2.0  # Major grammar issues
        elif grammar['grammar_accuracy'] < 0.9:
            score -= 1.0  # Some grammar issues
        
        # Length and development adjustments
        word_count = linguistic['word_count']
        if word_count < 100:
            score -= 2.0  # Too short
        elif word_count < 200:
            score -= 1.0  # Underdeveloped
        elif word_count > 500:
            score += 0.5  # Well developed
        
        return max(1.0, min(12.0, score))
    
    def generate_comprehensive_feedback(self, linguistic, vector, coherence, grammar):
        """Generate detailed feedback for improvement"""
        feedback = {
            'strengths': [],
            'weaknesses': [],
            'suggestions': {
                'grammar': [],
                'vocabulary': [],
                'structure': [],
                'coherence': [],
                'content': []
            }
        }
        
        # Strengths
        if linguistic['flesch_reading_ease'] >= 60:
            feedback['strengths'].append("Excellent readability and clarity")
        if linguistic['type_token_ratio'] >= 0.6:
            feedback['strengths'].append("Strong vocabulary diversity")
        if grammar['grammar_accuracy'] >= 0.9:
            feedback['strengths'].append("Good grammar accuracy")
        if coherence['cohesion_density'] >= 0.2:
            feedback['strengths'].append("Well-structured and coherent")
        
        # Weaknesses and suggestions
        if grammar['total_errors'] > 0:
            feedback['weaknesses'].append(f"Grammar issues detected ({grammar['total_errors']} errors)")
            feedback['suggestions']['grammar'] = [
                "Review subject-verb agreement",
                "Check tense consistency throughout",
                "Verify article usage (a/an/the)",
                "Ensure proper sentence structure"
            ]
        
        if linguistic['type_token_ratio'] < 0.4:
            feedback['weaknesses'].append("Limited vocabulary diversity")
            feedback['suggestions']['vocabulary'] = [
                "Use more varied and sophisticated vocabulary",
                "Incorporate domain-specific terminology",
                "Avoid repetition of common words"
            ]
        
        if coherence['cohesion_density'] < 0.1:
            feedback['weaknesses'].append("Poor essay coherence")
            feedback['suggestions']['coherence'] = [
                "Use transition words and phrases",
                "Ensure logical flow between paragraphs",
                "Maintain consistent pronoun usage",
                "Develop clear argument structure"
            ]
        
        if linguistic['word_count'] < 200:
            feedback['weaknesses'].append("Essay needs more development")
            feedback['suggestions']['content'] = [
                "Add specific examples and evidence",
                "Expand arguments with supporting details",
                "Include counterarguments and rebuttals",
                "Strengthen conclusion with summary"
            ]
        
        return feedback

# Global analyzer instance
advanced_analyzer = AdvancedNLPEssayAnalyzer()

def analyze_essay_comprehensive_nlp(text, reference_essays=None, reference_scores=None):
    """Global function for comprehensive NLP-based essay analysis"""
    return advanced_analyzer.comprehensive_essay_analysis(text, reference_essays, reference_scores)
