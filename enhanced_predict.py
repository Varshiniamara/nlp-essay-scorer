import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import textstat
import pandas as pd

class EnhancedEssayAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Common grammar patterns and their fixes
        self.grammar_patterns = [
            {
                'pattern': r'\bi\s+[a-z]',
                'error': 'Lowercase after "I"',
                'fix': 'Capitalize the first letter after "I"',
                'severity': 'minor'
            },
            {
                'pattern': r'\s{2,}',
                'error': 'Multiple spaces',
                'fix': 'Use single space between words',
                'severity': 'minor'
            },
            {
                'pattern': r'[.!?]{2,}',
                'error': 'Multiple punctuation',
                'fix': 'Use single punctuation mark',
                'severity': 'minor'
            },
            {
                'pattern': r'\b(a|an)\s+[aeiou]',
                'error': 'Article misuse',
                'fix': 'Check article usage (a vs an)',
                'severity': 'moderate'
            },
            {
                'pattern': r'\s+[,.]',
                'error': 'Space before punctuation',
                'fix': 'Remove space before punctuation',
                'severity': 'minor'
            },
            {
                'pattern': r'[,.]\s*[a-z]',
                'error': 'Lowercase after punctuation',
                'fix': 'Capitalize first letter after punctuation',
                'severity': 'minor'
            },
            {
                'pattern': r'\b(is|are|was|were)\s+\w+ing\b',
                'error': 'Incorrect verb tense',
                'fix': 'Check verb tense consistency',
                'severity': 'moderate'
            },
            {
                'pattern': r'\bthe\s+the\b',
                'error': 'Repeated "the"',
                'fix': 'Remove repeated words',
                'severity': 'minor'
            },
            {
                'pattern': r'\bvery\s+very\b',
                'error': 'Repeated "very"',
                'fix': 'Use stronger adjective instead',
                'severity': 'minor'
            },
            {
                'pattern': r'\bbecause\s+because\b',
                'error': 'Repeated "because"',
                'fix': 'Remove repeated words',
                'severity': 'minor'
            }
        ]
        
        # Style and structure issues
        self.style_patterns = [
            {
                'check': 'sentence_length',
                'description': 'Very long sentences',
                'threshold': 25,
                'fix': 'Break long sentences into shorter ones'
            },
            {
                'check': 'paragraph_length',
                'description': 'Very long paragraphs',
                'threshold': 150,
                'fix': 'Split long paragraphs'
            },
            {
                'check': 'word_repetition',
                'description': 'Repeated words',
                'threshold': 3,
                'fix': 'Use synonyms or rephrase'
            },
            {
                'check': 'vocabulary_diversity',
                'description': 'Low vocabulary diversity',
                'threshold': 0.3,
                'fix': 'Use more varied vocabulary'
            }
        ]
    
    def find_grammar_errors(self, text):
        """Find specific grammar errors with locations"""
        errors = []
        lines = text.split('\n')
        
        for i, pattern_info in enumerate(self.grammar_patterns):
            pattern = pattern_info['pattern']
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for match in matches:
                line_num = text[:match.start()].count('\n') + 1
                start_pos = match.start() - text[:match.start()].rfind('\n', 0, match.start())
                
                errors.append({
                    'type': 'grammar',
                    'error': pattern_info['error'],
                    'fix': pattern_info['fix'],
                    'severity': pattern_info['severity'],
                    'line': line_num,
                    'position': start_pos,
                    'context': self.get_context(text, match.start(), 30),
                    'match_text': match.group()
                })
        
        return errors
    
    def get_context(self, text, position, window_size=30):
        """Get context around a position in text"""
        start = max(0, position - window_size)
        end = min(len(text), position + window_size + len(text[position:position+20]))
        return text[start:end].strip()
    
    def analyze_style_issues(self, text):
        """Analyze style and structure issues"""
        issues = []
        sentences = sent_tokenize(text)
        words = text.lower().split()
        
        # Check sentence length
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) > 25:
                issues.append({
                    'type': 'style',
                    'issue': 'Very long sentence',
                    'fix': 'Break into shorter sentences',
                    'severity': 'moderate',
                    'line': i + 1,
                    'context': sentence[:100] + '...' if len(sentence) > 100 else sentence
                })
        
        # Check word repetition
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        for word, count in word_freq.items():
            if count >= 3:
                # Find positions of repeated words
                positions = []
                start = 0
                while True:
                    pos = text.lower().find(word, start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + len(word)
                
                issues.append({
                    'type': 'style',
                    'issue': f'Repeated word: "{word}"',
                    'fix': f'Use synonyms or rephrase "{word}"',
                    'severity': 'minor',
                    'positions': positions[:3],  # Show first 3 occurrences
                    'count': count
                })
        
        # Check vocabulary diversity
        unique_words = len(set([w for w in words if len(w) > 3]))
        total_words = len([w for w in words if len(w) > 3])
        vocab_diversity = unique_words / total_words if total_words > 0 else 0
        
        if vocab_diversity < 0.3:
            issues.append({
                'type': 'style',
                'issue': 'Low vocabulary diversity',
                'fix': 'Use more varied vocabulary',
                'severity': 'moderate',
                'metric': f'Diversity: {vocab_diversity:.2f}'
            })
        
        return issues
    
    def calculate_detailed_score(self, text, errors, style_issues):
        """Calculate score based on specific errors and issues"""
        base_score = 12.0  # Start from perfect score
        
        # Deduct points for grammar errors
        for error in errors:
            if error['severity'] == 'minor':
                base_score -= 0.5
            elif error['severity'] == 'moderate':
                base_score -= 1.0
            elif error['severity'] == 'major':
                base_score -= 2.0
        
        # Deduct points for style issues
        for issue in style_issues:
            if issue['severity'] == 'minor':
                base_score -= 0.3
            elif issue['severity'] == 'moderate':
                base_score -= 0.5
            elif issue['severity'] == 'major':
                base_score -= 1.0
        
        # Adjust for content quality
        words = text.split()
        if len(words) < 100:
            base_score -= 2.0  # Too short
        elif len(words) < 200:
            base_score -= 1.0  # Short
        elif len(words) > 500:
            base_score += 0.5  # Well developed
        
        return max(1.0, min(12.0, base_score))
    
    def generate_improvement_suggestions(self, errors, style_issues):
        """Generate specific improvement suggestions"""
        suggestions = []
        
        # Grammar-based suggestions
        grammar_errors = [e for e in errors if e['type'] == 'grammar']
        if grammar_errors:
            suggestions.append({
                'category': 'Grammar',
                'priority': 'high',
                'items': [
                    'Review and correct grammar errors',
                    'Check subject-verb agreement',
                    'Ensure proper punctuation usage',
                    'Verify article usage (a/an)'
                ]
            })
        
        # Style-based suggestions
        style_errors = [e for e in style_issues if e['type'] == 'style']
        if style_errors:
            suggestions.append({
                'category': 'Style & Structure',
                'priority': 'medium',
                'items': [
                    'Break down long sentences for clarity',
                    'Improve paragraph organization',
                    'Use more varied vocabulary',
                    'Avoid word repetition',
                    'Enhance sentence flow'
                ]
            })
        
        # Content-based suggestions
        suggestions.append({
            'category': 'Content Development',
            'priority': 'medium',
            'items': [
                'Add specific examples and evidence',
                'Develop stronger topic sentences',
                'Include transition words between ideas',
                'Expand with more supporting details',
                'Strengthen conclusion with summary'
            ]
        })
        
        return suggestions
    
    def analyze_essay_comprehensive(self, text):
        """Comprehensive essay analysis with mistake identification"""
        if not text or len(text.strip()) < 10:
            return {
                'score': 1.0,
                'error': 'Essay too short for meaningful analysis',
                'errors': [],
                'style_issues': [],
                'suggestions': [],
                'metrics': {}
            }
        
        # Find all errors
        grammar_errors = self.find_grammar_errors(text)
        style_issues = self.analyze_style_issues(text)
        
        # Calculate metrics
        words = text.split()
        sentences = sent_tokenize(text)
        
        metrics = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'char_count': len(text),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'vocab_diversity': len(set(words)) / len(words) if words else 0,
            'readability_score': textstat.flesch_reading_ease(text) if text else 0,
            'grammar_error_count': len(grammar_errors),
            'style_issue_count': len(style_issues),
            'complex_words': textstat.difficult_words(text) if text else 0
        }
        
        # Calculate score
        score = self.calculate_detailed_score(text, grammar_errors, style_issues)
        
        # Generate suggestions
        suggestions = self.generate_improvement_suggestions(grammar_errors, style_issues)
        
        return {
            'score': round(score, 1),
            'metrics': metrics,
            'errors': grammar_errors,
            'style_issues': style_issues,
            'suggestions': suggestions,
            'error': None
        }

# Global analyzer instance
analyzer = EnhancedEssayAnalyzer()

def analyze_essay_with_mistakes(text):
    """Global function for comprehensive essay analysis"""
    return analyzer.analyze_essay_comprehensive(text)
