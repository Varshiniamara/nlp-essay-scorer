
"""
Demo Essays Dataset for AES System
Contains 5 essays with different quality levels for demonstration
Matching the specific requirements for demo consistency
"""

DEMO_ESSAYS = {
    "highest": {
        "score_range": (9.0, 10.0),
        "grammar_score": 95,
        "vocabulary_score": 92,
        "coherence_score": 94,
        "readability_score": 93,
        "overall_score": 11.4, # Average of scores mapped to 12 scale
        "ai_recommendations": {
            "strengths": [
                "Good vocabulary usage",
                "Clear sentence structure",
                "Proper grammar"
            ],
            "weaknesses": [
                "Add more complex examples",
                "Use more analytical discussion",
                "Enhance critical thinking depth"
            ],
            "suggestions": [
                "Read academic essays",
                "Practice writing analytical paragraphs",
                "Use advanced transition words"
            ],
            "performance_insights": {
                "strengths": [
                    "Excellent grammar with minimal errors",
                    "Strong organization and logical flow",
                    "High-quality academic vocabulary"
                ],
                "areas_for_improvement": [
                    "Focus on improving overall essay quality by adding deeper analysis"
                ]
            }
        },
        "topic": "Technology in Education",
        "content": "Technology has significantly transformed the education system in recent years by providing innovative tools that enhance both teaching and learning experiences. In traditional classrooms, students relied primarily on textbooks and lectures, which sometimes limited their understanding of complex concepts. However, the introduction of digital platforms, virtual classrooms, and interactive simulations has made learning more engaging and accessible. Students can now access educational resources from anywhere, allowing them to study at their own pace and revisit difficult topics whenever necessary. Furthermore, teachers can monitor student progress using online assessment tools and provide personalized feedback to improve academic performance. Technology also encourages collaboration among students through discussion forums, group projects, and shared digital resources. Although the use of technology offers many benefits, it is important to maintain a balance between digital and traditional learning methods to ensure effective education."
    },
    
    "good": {
        "score_range": (7.0, 8.0),
        "grammar_score": 82,
        "vocabulary_score": 78,
        "coherence_score": 80,
        "readability_score": 79,
        "overall_score": 9.6,
        "ai_recommendations": {
            "strengths": [
                "Good vocabulary usage",
                "Clear sentence structure",
                "Proper grammar"
            ],
            "weaknesses": [
                "Add more supporting examples",
                "Use more advanced vocabulary",
                "Improve paragraph transitions"
            ],
            "suggestions": [
                "Read academic essays",
                "Practice writing longer paragraphs",
                "Use transition words"
            ],
            "performance_insights": {
                "strengths": [
                    "Good essay length and content development",
                    "Clear communication of ideas"
                ],
                "areas_for_improvement": [
                    "Focus on improving overall essay quality"
                ]
            }
        },
        "topic": "Online Learning",
        "content": "Online learning has become increasingly popular among students due to its flexibility and convenience. Many educational institutions provide digital learning platforms that allow students to attend classes remotely and access study materials at any time. This approach is especially useful for working professionals and students living in remote areas. Online learning also helps students develop technical skills that are important in the modern workplace. However, some students face challenges such as limited internet connectivity and lack of motivation during virtual classes. Teachers must therefore design engaging lessons and interactive activities to keep students interested. By combining technology with effective teaching methods, online education can provide valuable learning opportunities for students around the world."
    },
    
    "average": {
        "score_range": (5.0, 6.0),
        "grammar_score": 65,
        "vocabulary_score": 60,
        "coherence_score": 63,
        "readability_score": 62,
        "overall_score": 7.6,
        "ai_recommendations": {
            "strengths": [
                "Clear sentence structure",
                "Relevant topic discussion"
            ],
            "weaknesses": [
                "Use more varied vocabulary",
                "Add supporting examples",
                "Improve sentence complexity"
            ],
            "suggestions": [
                "Practice writing daily",
                "Use descriptive words",
                "Expand ideas with examples"
            ],
            "performance_insights": {
                "strengths": [
                    "Adequate essay length",
                    "Basic idea development"
                ],
                "areas_for_improvement": [
                    "Focus on improving overall essay quality"
                ]
            }
        },
        "topic": "Time Management",
        "content": "Time management is important for students because it helps them finish their work on time and reduce stress. Many students have different responsibilities such as attending classes, completing assignments, and preparing for exams. If students do not manage their time properly, they may feel overwhelmed and struggle to meet deadlines. Creating a daily schedule can help students organize their tasks and stay focused. However, some students find it difficult to follow a strict routine because they get distracted by social media or entertainment. With regular practice and discipline, students can improve their time management skills and achieve better academic performance."
    },
    
    "below_average": {
        "score_range": (3.0, 4.0),
        "grammar_score": 48,
        "vocabulary_score": 45,
        "coherence_score": 47,
        "readability_score": 46,
        "overall_score": 5.6,
        "ai_recommendations": {
            "strengths": [
                "Simple idea presentation",
                "Basic communication of topic"
            ],
            "weaknesses": [
                "Correct grammar mistakes",
                "Use better vocabulary",
                "Improve sentence structure"
            ],
            "suggestions": [
                "Practice grammar exercises",
                "Read English books",
                "Use proper punctuation"
            ],
            "performance_insights": {
                "strengths": [
                    "Basic essay structure present"
                ],
                "areas_for_improvement": [
                    "Focus on improving overall essay quality"
                ]
            }
        },
        "topic": "Social Media",
        "content": "Social media is very popular among students and many student use it everyday. It help students to talk with friends and share information but sometimes it create problems in their studies. Many students spend too much time on social media and forget to complete homework. Because of this their marks become low and teachers get worried. Students should control their social media usage and focus more on studies."
    },
    
    "lowest": {
        "score_range": (1.0, 2.0),
        "grammar_score": 33.3,
        "vocabulary_score": 33.3,
        "coherence_score": 33.3,
        "readability_score": 33.3,
        "overall_score": 4.0,
        "ai_recommendations": {
            "strengths": [
                "Good vocabulary usage",
                "Clear sentence structure",
                "Proper grammar"
            ],
            "weaknesses": [
                "Add more complex sentences",
                "Use advanced vocabulary",
                "Improve coherence"
            ],
            "suggestions": [
                "Read academic essays",
                "Practice writing daily",
                "Use transition words"
            ],
            "performance_insights": {
                "strengths": [
                    "Good essay length and content development"
                ],
                "areas_for_improvement": [
                    "Focus on improving overall essay quality"
                ]
            }
        },
        "topic": "Study Hard",
        "content": "students should study hard because education is important in life and it help them get good job many students not study properly they waste time playing games and watching videos teacher tell students to study but they not listen they fail in exams and feel sad students should work hard to become successful"
    }
}

GENERIC_PROFILES = [
    {
        "overall_score": 7.5,
        "grammar_score": 68.0,
        "vocabulary_score": 72.0,
        "coherence_score": 70.0,
        "readability_score": 65.0,
        "ai_recommendations": {
            "strengths": ["Clear main idea", "Moderate vocabulary diversity"],
            "weaknesses": ["Minor punctuation errors", "Simple sentence structures"],
            "suggestions": ["Use more complex transition words", "Expand on your examples"],
            "performance_insights": {
                "strengths": ["Consistent tone throughout the essay"],
                "areas_for_improvement": ["Focus on removing repetitive word choices"]
            }
        }
    },
    {
        "overall_score": 8.8,
        "grammar_score": 85.0,
        "vocabulary_score": 82.0,
        "coherence_score": 88.0,
        "readability_score": 80.0,
        "ai_recommendations": {
            "strengths": ["Strong argumentation", "Sophisticated word choice"],
            "weaknesses": ["Occasional flow issues between paragraphs"],
            "suggestions": ["Bridge your ideas more naturally"],
            "performance_insights": {
                "strengths": ["Excellent use of academic terminology"],
                "areas_for_improvement": ["Vary your sentence lengths further"]
            }
        }
    },
    {
        "overall_score": 5.2,
        "grammar_score": 50.0,
        "vocabulary_score": 48.0,
        "coherence_score": 55.0,
        "readability_score": 42.0,
        "ai_recommendations": {
            "strengths": ["Direct response to the prompt"],
            "weaknesses": ["Frequent spelling mistakes", "Fragmented sentences"],
            "suggestions": ["Read your work aloud to catch errors", "Study basic grammar rules"],
            "performance_insights": {
                "strengths": ["Topic relevance is maintained"],
                "areas_for_improvement": ["Practice basic sentence structuring"]
            }
        }
    }
]

def get_demo_essay(quality_level):
    """Get demo essay by quality level"""
    return DEMO_ESSAYS.get(quality_level, DEMO_ESSAYS["average"])

def get_all_demo_essays():
    """Get all demo essays for testing"""
    return DEMO_ESSAYS

def analyze_demo_essay_quality(essay_text):
    """Simple quality analysis for demo purposes"""
    text = essay_text.strip().lower()
    for level, data in DEMO_ESSAYS.items():
        if text.startswith(data['content'].strip().lower()[:50]):
            return level, data['overall_score']
    
    word_count = len(essay_text.split())
    if word_count > 100:
        return "average", 7.6
    return "lowest", 4.0
