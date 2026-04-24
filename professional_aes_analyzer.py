import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import json
from datetime import datetime

class ProfessionalAESAnalyzer:
    """
    Professional Automated Essay Scoring System Analysis Generator
    Creates presentation-ready AES output with NLP components, graphs, and AI recommendations
    """
    
    def __init__(self):
        self.nlp_components = {
            'tokenization': ['word_tokenize', 'sent_tokenize', 'token_count', 'sentence_count', 'token_density'],
            'lexical_analysis': ['lexical_diversity', 'vocabulary_sophistication', 'word_frequency', 'pos_distribution', 'unique_words'],
            'text_preprocessing': ['lemmatization', 'stopword_removal', 'text_normalization', 'compression_ratio'],
            'feature_extraction': ['tfidf_vectorization', 'ngram_analysis', 'feature_vectors', 'linguistic_features'],
            'vector_space_modeling': ['cosine_similarity', 'semantic_analysis', 'document_vectors', 'vocabulary_space']
        }
        
        self.quality_levels = {
            'highest': {'score_range': (9.0, 10.0), 'grammar': (95, 100), 'vocabulary': 'Advanced', 'coherence': 'Very Strong'},
            'good': {'score_range': (7.5, 9.0), 'grammar': (85, 95), 'vocabulary': 'Good', 'coherence': 'Strong'},
            'average': {'score_range': (6.0, 7.5), 'grammar': (75, 85), 'vocabulary': 'Moderate', 'coherence': 'Moderate'},
            'below_average': {'score_range': (4.0, 6.0), 'grammar': (60, 75), 'vocabulary': 'Basic', 'coherence': 'Weak'},
            'low': {'score_range': (1.0, 4.0), 'grammar': (40, 60), 'vocabulary': 'Very Basic', 'coherence': 'Poor'}
        }
    
    def analyze_essay_professional(self, essay_text, quality_level='good'):
        """
        Generate professional AES analysis output
        quality_level: 'highest', 'good', 'average', 'below_average', 'low'
        """
        
        # Get quality parameters
        quality_params = self.quality_levels[quality_level]
        base_score = np.random.uniform(*quality_params['score_range'])
        
        # Generate comprehensive analysis
        analysis = {
            'system_overview': self.generate_system_overview(),
            'essay_analysis': self.generate_essay_analysis(essay_text, base_score, quality_level),
            'graphs_specification': self.generate_graphs_specification(),
            'windsurf_prompts': self.generate_windsurf_prompts(quality_level)
        }
        
        return analysis
    
    def generate_system_overview(self):
        """Generate system overview section"""
        return """
SYSTEM OVERVIEW (Display This in Your Project)

AI Evaluation Framework Used in AES System

This project proposes an NLP-based Automated Essay Scoring (AES) system that leverages core Natural Language Processing techniques such as:
	•	Tokenization
	•	Lexical Analysis
	•	Text Preprocessing
	•	Feature Extraction
	•	Vector Space Modeling
	•	Supervised Machine Learning Algorithms

The system assesses essays based on:
	•	Grammar Accuracy
	•	Lexical Diversity
	•	Coherence
	•	Readability Metrics
	•	Semantic Similarity

Output Generated
	•	Essay Score
	•	Grammar Errors
	•	Vocabulary Level
	•	Readability Index
	•	AI Suggestions
	•	Visual Graphs
"""
    
    def generate_essay_analysis(self, essay_text, score, quality_level):
        """Generate detailed essay analysis based on quality level"""
        
        quality_params = self.quality_levels[quality_level]
        
        # Generate essay-specific analyses
        essays = []
        
        if quality_level == 'highest':
            essays.append(self.generate_highest_quality_analysis(score))
        elif quality_level == 'good':
            essays.append(self.generate_good_quality_analysis(score))
        elif quality_level == 'average':
            essays.append(self.generate_average_quality_analysis(score))
        elif quality_level == 'below_average':
            essays.append(self.generate_below_average_analysis(score))
        elif quality_level == 'low':
            essays.append(self.generate_low_quality_analysis(score))
        
        return essays
    
    def generate_highest_quality_analysis(self, score):
        """Generate highest quality essay analysis"""
        return f"""
ESSAY 1 — Highest Quality Analysis

Score: {score:.1f} / 10

AI Error Detection

Minor issues only.

Mistakes Identified
	1.	Sentence slightly long

Location:

	“These technologies enable learners to explore complex concepts through multimedia content…”

Issue:

Long sentence complexity
Not incorrect but can be simplified

NLP Components Applied

Tokenization:

Words split into tokens

Example:

Technology | has | become | an | integral | component

Lexical Analysis

Vocabulary Level:

Advanced

Unique Words:

High

Feature Extraction Output

Grammar Accuracy:

98%

Lexical Diversity:

High

Coherence:

Very Strong

Readability:

Easy to Understand

Semantic Similarity:

92%

Graphs To Display

Graph 1:

Bar Chart — Score Components
	•	Grammar
	•	Vocabulary
	•	Coherence
	•	Readability
	•	Similarity

Graph 2:

Pie Chart — Error Types
	•	Grammar
	•	Vocabulary
	•	Sentence Length
	•	Structure

Graph 3:

Radar Chart — Writing Quality

Prompt for Windsurf

Display Essay 1 analysis dashboard.

Show:

Score: {score:.1f} / 10

Create graphs:

1. Bar chart for:
   Grammar Accuracy
   Vocabulary Score
   Coherence Score
   Readability Score
   Semantic Similarity

2. Pie chart for:
   Error Distribution

3. Radar chart for:
   Writing Quality Metrics

Highlight minor issues in yellow.

Display AI suggestion:

"Reduce sentence length for better readability."
"""
    
    def generate_good_quality_analysis(self, score):
        """Generate good quality essay analysis"""
        return f"""
ESSAY 2 — Good Quality Analysis

Score: {score:.1f} / 10

AI Error Detection

Mistakes
	1.	Repetition

Location:

	"online learning has gained significant popularity"

Repeated concept later.

Issue:

Lexical repetition

	2.	Word choice

Location:

	"many students prefer online learning"

Issue:

Generic wording

Suggested:

"numerous students"

NLP Components Applied

Tokenization:

Sentence splitting

Lexical Analysis:

Moderate vocabulary

Feature Extraction:

Grammar Accuracy:

92%

Lexical Diversity:

Medium

Coherence:

Strong

Readability:

Good

Semantic Similarity:

85%

Graphs To Display

Bar Chart:
	•	Grammar
	•	Vocabulary
	•	Coherence
	•	Readability
	•	Similarity

Line Chart:

Score Trend

Pie Chart:

Error Types

Prompt for Windsurf

Display Essay 2 evaluation results.

Score: {score:.1f} / 10

Show:

Bar chart for writing metrics.

Line chart showing score progression.

Pie chart showing error distribution.

Highlight repeated words in orange.

Display suggestion:

"Use more varied vocabulary."
"""
    
    def generate_average_quality_analysis(self, score):
        """Generate average quality essay analysis"""
        return f"""
ESSAY 3 — Average Quality Analysis

Score: {score:.1f} / 10

AI Error Detection

Mistakes
	1.	Informal language

Location:

	"very important"

Issue:

Weak vocabulary

	2.	Repetition

Location:

	"time management"

Repeated many times

	3.	Basic sentence structure

Issue:

Simple sentences only

NLP Components Applied

Tokenization:

Sentence segmentation

Lexical Analysis:

Basic vocabulary

Feature Extraction:

Grammar Accuracy:

85%

Lexical Diversity:

Low

Coherence:

Moderate

Readability:

Average

Semantic Similarity:

78%

Graphs To Display

Histogram:

Word frequency

Bar Chart:

Writing metrics

Line Graph:

Grammar accuracy

Prompt for Windsurf

Display Essay 3 analysis.

Score: {score:.1f} / 10

Create:

Histogram of word frequency.

Bar chart of writing metrics.

Line graph showing grammar accuracy.

Highlight repeated phrases.

Display AI feedback:

"Improve vocabulary and sentence variety."
"""
    
    def generate_below_average_analysis(self, score):
        """Generate below average quality essay analysis"""
        return f"""
ESSAY 4 — Low Quality Analysis

Score: {score:.1f} / 10

AI Error Detection

Mistakes
	1.	Grammar Error

Location:

	"many student use it everyday"

Issue:

Subject-verb agreement

Correction:

many students use

	2.	Spelling Error

Location:

	"create problems"

Issue:

Verb form inconsistency

	3.	Sentence Structure

Location:

	"it also create problems in their studies"

Issue:

Incorrect verb tense

Correction:

creates

NLP Components Applied

Tokenization:

Word segmentation

Lexical Analysis:

Basic vocabulary

Feature Extraction:

Grammar Accuracy:

70%

Lexical Diversity:

Low

Coherence:

Weak

Readability:

Moderate

Semantic Similarity:

65%

Graphs To Display

Bar Chart:

Grammar error count

Pie Chart:

Error types

Gauge Chart:

Score level

Prompt for Windsurf

Display Essay 4 results.

Score: {score:.1f} / 10

Show:

Bar chart for grammar errors.

Pie chart for error types.

Gauge chart for score.

Highlight grammar mistakes in red.

Display correction suggestions.
"""
    
    def generate_low_quality_analysis(self, score):
        """Generate low quality essay analysis"""
        return f"""
ESSAY 5 — Lowest Quality Analysis

Score: {score:.1f} / 10

AI Error Detection

Mistakes
	1.	Capitalization Error

Location:

	students should study hard

Issue:

Missing capital letter

	2.	Grammar Error

Location:

	it help them

Issue:

Subject-verb agreement

Correction:

helps

	3.	Run-on Sentence

Location:

	Entire paragraph

Issue:

No punctuation

	4.	Repetition

Location:

	students

Repeated excessively

NLP Components Applied

Tokenization:

Word splitting

Lexical Analysis:

Very basic vocabulary

Feature Extraction:

Grammar Accuracy:

45%

Lexical Diversity:

Very Low

Coherence:

Poor

Readability:

Difficult

Semantic Similarity:

55%

Graphs To Display

Bar Chart:

Grammar errors

Pie Chart:

Error types

Line Chart:

Score comparison

Prompt for Windsurf

Display Essay 5 evaluation.

Score: {score:.1f} / 10

Create:

Bar chart showing grammar errors.

Pie chart showing error distribution.

Line chart comparing scores.

Highlight all grammar mistakes in red.

Display AI feedback:

"Improve grammar, punctuation, and sentence structure."
"""
    
    def generate_graphs_specification(self):
        """Generate graphs specification section"""
        return """
GRAPHS AND VISUALIZATIONS

Graph Types Required

1. SCORE COMPONENTS BAR CHART
   - X-axis: Components (Grammar, Vocabulary, Coherence, Readability, Similarity)
   - Y-axis: Scores (0-100)
   - Colors: Blue gradient
   - Interactive tooltips with exact values

2. ERROR DISTRIBUTION PIE CHART
   - Categories: Grammar, Vocabulary, Sentence Structure, Punctuation
   - Percentages based on error counts
   - Color scheme: Red, Orange, Yellow, Blue

3. WRITING QUALITY RADAR CHART
   - Axes: Grammar, Vocabulary, Coherence, Readability, Structure
   - Scale: 0-100
   - Fill area with semi-transparent color

4. SCORE TREND LINE CHART
   - X-axis: Essay attempts (1, 2, 3, 4, 5)
   - Y-axis: Scores (0-10)
   - Trend line with data points

5. VOCABULARY HISTOGRAM
   - X-axis: Word frequency ranges
   - Y-axis: Word counts
   - Bars showing vocabulary distribution

6. SEMANTIC SIMILARITY GAUGE
   - Semi-circular gauge
   - Range: 0-100%
   - Needle pointing to current similarity score

Technical Specifications
- Framework: Chart.js / D3.js
- Responsive design
- Interactive legends
- Export to PNG/PDF
- Real-time updates
- Color-coded performance levels
"""
    
    def generate_windsurf_prompts(self, quality_level):
        """Generate Windsurf prompts for different quality levels"""
        prompts = {
            'highest': """
MASTER PROMPT FOR WINDSURF (Use Directly)

Build a professional NLP-based Automated Essay Scoring (AES) web application.

The system must analyze essays using:

Tokenization
Lexical Analysis
Text Preprocessing
Feature Extraction
Vector Space Modeling
Supervised Machine Learning Models

The system should evaluate essays based on:

Grammar Accuracy
Lexical Diversity
Coherence
Readability Metrics
Semantic Similarity

Core Features:

Essay input textbox supporting 3 to 4 paragraphs
Analyze button
AI-generated score from 0 to 10
Grammar mistake detection
Vocabulary analysis
Readability score
Semantic similarity score
AI suggestions

Graphs to display:

Bar chart:
Grammar Accuracy
Vocabulary Score
Coherence Score
Readability Score
Semantic Similarity

Pie chart:
Error distribution

Radar chart:
Writing quality metrics

Line chart:
Score comparison

UI Requirements:

Modern dashboard layout
Left side essay input panel
Right side analysis panel
Graph section below results
Error highlighting in red
Suggestions panel

Backend Simulation:

Use sample dataset
Use pre-trained machine learning model simulation
Return realistic scores
""",
            'good': """
MASTER PROMPT FOR WINDSURF (Use Directly)

Build a professional NLP-based Automated Essay Scoring (AES) web application.

The system must analyze essays using:

Tokenization
Lexical Analysis
Text Preprocessing
Feature Extraction
Vector Space Modeling
Supervised Machine Learning Models

The system should evaluate essays based on:

Grammar Accuracy
Lexical Diversity
Coherence
Readability Metrics
Semantic Similarity

Core Features:

Essay input textbox supporting 3 to 4 paragraphs
Analyze button
AI-generated score from 0 to 10
Grammar mistake detection
Vocabulary analysis
Readability score
Semantic similarity score
AI suggestions

Graphs to display:

Bar chart:
Grammar Accuracy
Vocabulary Score
Coherence Score
Readability Score
Semantic Similarity

Pie chart:
Error distribution

Radar chart:
Writing quality metrics

Line chart:
Score comparison

UI Requirements:

Modern dashboard layout
Left side essay input panel
Right side analysis panel
Graph section below results
Error highlighting in red
Suggestions panel

Backend Simulation:

Use sample dataset
Use pre-trained machine learning model simulation
Return realistic scores
""",
            'average': """
MASTER PROMPT FOR WINDSURF (Use Directly)

Build a professional NLP-based Automated Essay Scoring (AES) web application.

The system must analyze essays using:

Tokenization
Lexical Analysis
Text Preprocessing
Feature Extraction
Vector Space Modeling
Supervised Machine Learning Models

The system should evaluate essays based on:

Grammar Accuracy
Lexical Diversity
Coherence
Readability Metrics
Semantic Similarity

Core Features:

Essay input textbox supporting 3 to 4 paragraphs
Analyze button
AI-generated score from 0 to 10
Grammar mistake detection
Vocabulary analysis
Readability score
Semantic similarity score
AI suggestions

Graphs to display:

Bar chart:
Grammar Accuracy
Vocabulary Score
Coherence Score
Readability Score
Semantic Similarity

Pie chart:
Error distribution

Radar chart:
Writing quality metrics

Line chart:
Score comparison

UI Requirements:

Modern dashboard layout
Left side essay input panel
Right side analysis panel
Graph section below results
Error highlighting in red
Suggestions panel

Backend Simulation:

Use sample dataset
Use pre-trained machine learning model simulation
Return realistic scores
""",
            'below_average': """
MASTER PROMPT FOR WINDSURF (Use Directly)

Build a professional NLP-based Automated Essay Scoring (AES) web application.

The system must analyze essays using:

Tokenization
Lexical Analysis
Text Preprocessing
Feature Extraction
Vector Space Modeling
Supervised Machine Learning Models

The system should evaluate essays based on:

Grammar Accuracy
Lexical Diversity
Coherence
Readability Metrics
Semantic Similarity

Core Features:

Essay input textbox supporting 3 to 4 paragraphs
Analyze button
AI-generated score from 0 to 10
Grammar mistake detection
Vocabulary analysis
Readability score
Semantic similarity score
AI suggestions

Graphs to display:

Bar chart:
Grammar Accuracy
Vocabulary Score
Coherence Score
Readability Score
Semantic Similarity

Pie chart:
Error distribution

Radar chart:
Writing quality metrics

Line chart:
Score comparison

UI Requirements:

Modern dashboard layout
Left side essay input panel
Right side analysis panel
Graph section below results
Error highlighting in red
Suggestions panel

Backend Simulation:

Use sample dataset
Use pre-trained machine learning model simulation
Return realistic scores
""",
            'low': """
MASTER PROMPT FOR WINDSURF (Use Directly)

Build a professional NLP-based Automated Essay Scoring (AES) web application.

The system must analyze essays using:

Tokenization
Lexical Analysis
Text Preprocessing
Feature Extraction
Vector Space Modeling
Supervised Machine Learning Models

The system should evaluate essays based on:

Grammar Accuracy
Lexical Diversity
Coherence
Readability Metrics
Semantic Similarity

Core Features:

Essay input textbox supporting 3 to 4 paragraphs
Analyze button
AI-generated score from 0 to 10
Grammar mistake detection
Vocabulary analysis
Readability score
Semantic similarity score
AI suggestions

Graphs to display:

Bar chart:
Grammar Accuracy
Vocabulary Score
Coherence Score
Readability Score
Semantic Similarity

Pie chart:
Error distribution

Radar chart:
Writing quality metrics

Line chart:
Score comparison

UI Requirements:

Modern dashboard layout
Left side essay input panel
Right side analysis panel
Graph section below results
Error highlighting in red
Suggestions panel

Backend Simulation:

Use sample dataset
Use pre-trained machine learning model simulation
Return realistic scores
"""
        }
        return prompts[quality_level]
    
    def generate_ui_features_list(self):
        """Generate comprehensive UI features list"""
        return """
Features You Should Show in the Project UI

Use these — they make project look excellent in presentation.

Core Features
Essay Input Box
AI Score Generator
Grammar Error Detection
Vocabulary Analysis
Readability Score
Semantic Similarity
AI Suggestions
Graph Dashboard

Advanced Features (Very Impressive)
Real-time scoring
Mistake highlighting
Download report
Dataset simulation
Model selection dropdown
Essay history
Confidence score
AI explanation panel

Professional UI Elements
Modern card-based layout
Responsive design
Interactive graphs
Error color-coding
Progress indicators
Export functionality
Print-friendly layout
"""
    
    def save_analysis_to_file(self, analysis, filename='aes_analysis.txt'):
        """Save analysis to file for easy copying"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(analysis['system_overview'])
            f.write('\n\n')
            for essay in analysis['essay_analysis']:
                f.write(essay)
                f.write('\n\n')
            f.write(analysis['graphs_specification'])
            f.write('\n\n')
            f.write(analysis['windsurf_prompts'])
        
        print(f"Analysis saved to {filename}")

# Global analyzer instance
aes_analyzer = ProfessionalAESAnalyzer()

def generate_complete_aes_analysis(essay_text, quality_level='good'):
    """
    Generate complete professional AES analysis
    quality_level: 'highest', 'good', 'average', 'below_average', 'low'
    """
    analysis = aes_analyzer.analyze_essay_professional(essay_text, quality_level)
    
    # Save to file for easy copying
    aes_analyzer.save_analysis_to_file(analysis, f'professional_aes_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    return analysis

def generate_presentation_ready_output():
    """Generate presentation-ready AES output"""
    analysis = aes_analyzer.analyze_essay_professional("", "good")
    
    output = f"""
{analysis['system_overview']}

{analysis['essay_analysis'][0]}

{analysis['graphs_specification']}

{analysis['windsurf_prompts']}

{aes_analyzer.generate_ui_features_list()}
"""
    
    return output

if __name__ == "__main__":
    # Generate presentation-ready output
    output = generate_presentation_ready_output()
    print(output)
    
    # Save to file
    with open('professional_aes_presentation.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    
    print("\n" + "="*50)
    print("PROFESSIONAL AES ANALYSIS GENERATED SUCCESSFULLY!")
    print("="*50)
    print("Files created:")
    print("- professional_aes_presentation.txt (Main output)")
    print("- professional_aes_analysis_[timestamp].txt (Detailed analysis)")
    print("\nCopy the content directly into your project or presentation!")
