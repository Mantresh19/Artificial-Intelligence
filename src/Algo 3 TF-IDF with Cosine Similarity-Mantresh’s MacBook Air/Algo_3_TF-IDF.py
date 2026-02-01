# TF-IDF with Cosine Similarity FAQ Chatbot for University Student Support
# Author: Kirtan Karkar
# COMP1827 Introduction to Artificial Intelligence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# SECTION 1: FAQ KNOWLEDGE BASE
# ============================================================================

faq_knowledge_base = {
    "deadlines": {
        "questions": [
            "When is the deadline?",
            "What is the due date?",
            "When do I need to submit?",
            "What's the submission date?",
            "When is it due?",
            "Final submission date?",
            "Hand in date?",
            "When should I submit my work?",
            "Deadline for presentation?",
            "Deadline for report?"
        ],
        "response": "The group presentation is on Monday, 1 December 2025. The individual report is due on Wednesday, 3 December 2025 at 5 PM.",
        "category": "assessment"
    },
    "late_submission": {
        "questions": [
            "Can I submit late?",
            "What if I miss the deadline?",
            "Late submission penalty?",
            "Extension request?",
            "What happens if I submit after deadline?",
            "How to apply for extension?",
            "Late work policy?",
            "Missed deadline help?",
            "Submit after due date?",
            "Extenuating circumstances?"
        ],
        "response": "Late submissions are subject to university penalties. Check the coursework regulations at http://www2.gre.ac.uk/current-students/regs or apply for Extenuating Circumstances if you have valid reasons.",
        "category": "assessment"
    },
    "module_leader": {
        "questions": [
            "Who is the module leader?",
            "Who teaches this course?",
            "Course instructor?",
            "Who is the lecturer?",
            "Professor name?",
            "Teacher for COMP1827?",
            "Who runs this module?",
            "Contact for module leader?",
            "Main lecturer?",
            "Course leader name?"
        ],
        "response": "The module leader for COMP1827 is Dr. Mohammad M. al-Rifaie.",
        "category": "module_info"
    },
    "module_name": {
        "questions": [
            "What is this module called?",
            "Module name?",
            "Course title?",
            "What module is this?",
            "COMP1827 name?",
            "Full module name?",
            "What course is this?",
            "Module code?",
            "Course name?",
            "What is COMP1827?"
        ],
        "response": "This module is COMP1827 Introduction to Artificial Intelligence.",
        "category": "module_info"
    },
    "submission_format": {
        "questions": [
            "How do I submit?",
            "Submission format?",
            "What format for report?",
            "PDF or Word?",
            "How to upload?",
            "LaTeX requirement?",
            "File format?",
            "Submission method?",
            "Where to submit?",
            "Upload instructions?"
        ],
        "response": "You must submit a single PDF document via Moodle. The report should be generated from LaTeX using the provided template. Make sure files are virus-free and not password protected.",
        "category": "submission"
    },
    "word_count": {
        "questions": [
            "How many words?",
            "Word limit?",
            "Word count requirement?",
            "Length of report?",
            "Maximum words?",
            "Minimum words?",
            "Report length?",
            "How long should report be?",
            "Word count for report?",
            "Number of words needed?"
        ],
        "response": "The individual report should be 1,500 words including references.",
        "category": "submission"
    },
    "marking_criteria": {
        "questions": [
            "How is it marked?",
            "Marking scheme?",
            "Grading criteria?",
            "Assessment criteria?",
            "How will it be graded?",
            "What are the marks for?",
            "Marking breakdown?",
            "Grade distribution?",
            "How to get good marks?",
            "What gets marks?"
        ],
        "response": "The report is marked on: Understanding of Problem Domain (30%), Development and Implementation (50%), Conclusions and Critical Review (10%), and Writing Quality with referencing (10%).",
        "category": "marking"
    },
    "presentation_marks": {
        "questions": [
            "Presentation marks?",
            "How is presentation graded?",
            "Demo assessment?",
            "Group presentation marking?",
            "What's the presentation worth?",
            "Presentation percentage?",
            "How much is demo?",
            "Group work marks?",
            "Presentation criteria?",
            "Demo grading?"
        ],
        "response": "The group presentation is worth 50% of total marks. It is assessed on: Understanding of Problem Domain (30%), Development and Implementation (50%), and Conclusions and Critical Review (20%).",
        "category": "marking"
    },
    "grade_boundaries": {
        "questions": [
            "What grade will I get?",
            "First class boundary?",
            "Pass mark?",
            "Grade thresholds?",
            "What percentage for 2:1?",
            "How to get first class?",
            "Distinction marks?",
            "Fail boundary?",
            "Grade percentages?",
            "Classification boundaries?"
        ],
        "response": "Grade boundaries: 80%+ is exceptional, 70-79% shows high standard, 60-69% demonstrates clear awareness, 50-59% is mainly descriptive, 40-49% is superficial, below 40% shows little evidence of criteria.",
        "category": "marking"
    },
    "group_size": {
        "questions": [
            "Group size?",
            "How many in group?",
            "Team members?",
            "How many people per group?",
            "Group composition?",
            "Number of group members?",
            "Team size requirement?",
            "How big is group?",
            "Members in team?",
            "Group of how many?"
        ],
        "response": "Students should work in groups of 4 people. All members should contribute equally to the group work.",
        "category": "group_work"
    },
    "group_work": {
        "questions": [
            "What is group work?",
            "Team responsibilities?",
            "Group tasks?",
            "Collaboration requirements?",
            "What do we do as group?",
            "Group activities?",
            "Team project work?",
            "Working together how?",
            "Group contribution?",
            "Teamwork expectations?"
        ],
        "response": "Group activities include: discussing topics, identifying individual research areas, sharing progress, and preparing the presentation together. The presentation mark is shared equally among group members.",
        "category": "group_work"
    },
    "plagiarism": {
        "questions": [
            "What is plagiarism?",
            "Can I copy?",
            "Cheating policy?",
            "Academic misconduct?",
            "Citation requirements?",
            "Referencing rules?",
            "Copying penalty?",
            "How to avoid plagiarism?",
            "Academic integrity?",
            "What counts as cheating?"
        ],
        "response": "Plagiarism includes copying from web/books without referencing, submitting joint work as individual, or copying other students. All material must be properly referenced. Work is checked for plagiarism.",
        "category": "academic_integrity"
    },
    "lsepi": {
        "questions": [
            "What is LSEPI?",
            "Legal issues?",
            "Ethical considerations?",
            "Social impact?",
            "Professional issues?",
            "Ethics in AI?",
            "Legal requirements?",
            "Social responsibility?",
            "Professional standards?",
            "LSEPI meaning?"
        ],
        "response": "You must consider Legal, Social, Ethical and Professional Issues (LSEPI) in your work. This includes thinking about how AI systems impact society, privacy concerns, and professional responsibilities.",
        "category": "requirements"
    },
    "contact": {
        "questions": [
            "How to contact?",
            "Get help?",
            "Ask questions where?",
            "Support available?",
            "Email address?",
            "Who to contact?",
            "Help desk?",
            "Where to ask?",
            "Contact information?",
            "Need assistance?"
        ],
        "response": "For questions, contact your lecturer via university email, check Moodle announcements, or visit during lab sessions for topic approval.",
        "category": "support"
    },
    "moodle": {
        "questions": [
            "Where is Moodle?",
            "Course page location?",
            "Online materials?",
            "Moodle link?",
            "Access course materials?",
            "Find resources?",
            "Course website?",
            "Learning platform?",
            "Virtual learning environment?",
            "Where to find materials?"
        ],
        "response": "All coursework materials, submission links, and announcements are available on the COMP1827 Moodle page.",
        "category": "support"
    },
    "learning_outcomes": {
        "questions": [
            "What will I learn?",
            "Course objectives?",
            "Learning goals?",
            "Skills gained?",
            "What's covered?",
            "Module outcomes?",
            "What do I achieve?",
            "Course aims?",
            "Learning targets?",
            "What's the purpose?"
        ],
        "response": "By completing this course you will: understand AI methods and their applications, identify knowledge representation models and search algorithms, and be aware of philosophical and ethical issues in AI.",
        "category": "module_info"
    },
    "workload": {
        "questions": [
            "How much time?",
            "Hours needed?",
            "Time commitment?",
            "How long to complete?",
            "Expected effort?",
            "Study hours?",
            "Workload estimate?",
            "Time required?",
            "How many hours?",
            "Effort needed?"
        ],
        "response": "This coursework should take approximately 50 hours for an average student who is up-to-date with tutorial work.",
        "category": "module_info"
    }
}


# ============================================================================
# SECTION 2: TF-IDF VECTORIZER CONFIGURATION
# ============================================================================

class TFIDFChatbot:
    def __init__(self, faq_data,
                 min_df=1,
                 max_df=0.8,
                 ngram_range=(1, 2),
                 threshold=0.3):
        """
        Initialize TF-IDF based FAQ chatbot.

        Args:
            faq_data: Dictionary containing FAQ questions and responses
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms (ignore common words)
            ngram_range: Range for n-grams (1,2 means unigrams and bigrams)
            threshold: Minimum cosine similarity threshold for accepting match
        """
        self.faq_data = faq_data
        self.threshold = threshold

        # Initialize vectorizer with preprocessing
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words='english',
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Words with 2+ letters
        )

        # Build document corpus
        self.questions = []
        self.responses = []
        self.categories = []
        self.intent_names = []

        for intent_name, intent_data in faq_data.items():
            for question in intent_data['questions']:
                self.questions.append(question)
                self.responses.append(intent_data['response'])
                self.categories.append(intent_data['category'])
                self.intent_names.append(intent_name)

        # Fit TF-IDF model on all questions
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

        # Performance tracking
        self.interaction_log = []

    def preprocess_query(self, query):
        """Basic preprocessing of user query."""
        query = query.lower().strip()
        query = re.sub(r'[^\w\s]', '', query)  # Remove punctuation
        return query

    def find_best_match(self, user_query):
        """
        Find best matching FAQ using TF-IDF and cosine similarity.

        Returns:
            tuple: (response, similarity_score, intent, top_matches)
        """
        # Preprocess query
        processed_query = self.preprocess_query(user_query)

        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])

        # Calculate cosine similarity with all FAQ questions
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # Get top 5 matches for analysis
        top_indices = np.argsort(similarities)[-5:][::-1]
        top_matches = [
            {
                'question': self.questions[idx],
                'intent': self.intent_names[idx],
                'similarity': similarities[idx],
                'category': self.categories[idx]
            }
            for idx in top_indices
        ]

        # Best match
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        if best_score >= self.threshold:
            response = self.responses[best_idx]
            intent = self.intent_names[best_idx]
        else:
            response = None
            intent = None

        return response, best_score, intent, top_matches

    def get_response(self, user_query):
        """Get chatbot response for user query."""
        if not user_query.strip():
            return "Please ask a question about the coursework or module."

        response, score, intent, top_matches = self.find_best_match(user_query)

        # Log interaction
        self.log_interaction(user_query, response, score, intent)

        if response:
            return response
        else:
            return ("I'm not confident about that question. Try asking about: "
                    "deadlines, submission format, marking criteria, group work, "
                    "plagiarism, or module information.")

    def log_interaction(self, query, response, score, intent):
        """Log interaction for performance analysis."""
        self.interaction_log.append({
            'query': query,
            'response': response,
            'similarity_score': score,
            'intent': intent,
            'successful': score >= self.threshold
        })

    def get_performance_stats(self):
        """Calculate performance statistics."""
        if not self.interaction_log:
            return {'message': 'No interactions logged yet.'}

        total = len(self.interaction_log)
        successful = sum(1 for log in self.interaction_log if log['successful'])

        successful_scores = [log['similarity_score']
                             for log in self.interaction_log
                             if log['successful']]

        avg_similarity = (sum(successful_scores) / len(successful_scores)
                          if successful_scores else 0)

        # Intent frequency
        intent_counts = defaultdict(int)
        for log in self.interaction_log:
            if log['intent']:
                intent_counts[log['intent']] += 1

        # Failed queries
        failed_queries = [log['query']
                          for log in self.interaction_log
                          if not log['successful']]

        return {
            'total_interactions': total,
            'successful_matches': successful,
            'failed_matches': total - successful,
            'success_rate': round((successful / total) * 100, 1),
            'avg_similarity': round(avg_similarity, 3),
            'intent_frequency': dict(intent_counts),
            'failed_queries': failed_queries
        }

    def display_stats(self):
        """Display performance statistics."""
        stats = self.get_performance_stats()

        if 'message' in stats:
            print(stats['message'])
            return

        print("\n" + "=" * 60)
        print("TF-IDF CHATBOT PERFORMANCE STATISTICS")
        print("=" * 60)
        print(f"Total interactions: {stats['total_interactions']}")
        print(f"Successful matches: {stats['successful_matches']}")
        print(f"Failed matches: {stats['failed_matches']}")
        print(f"Success rate: {stats['success_rate']}%")
        print(f"Average similarity: {stats['avg_similarity']}")

        if stats['intent_frequency']:
            print("\nMost common intents:")
            sorted_intents = sorted(stats['intent_frequency'].items(),
                                    key=lambda x: x[1], reverse=True)
            for intent, count in sorted_intents[:5]:
                print(f"  - {intent}: {count} times")

        if stats['failed_queries']:
            print("\nQueries that failed to match:")
            for query in stats['failed_queries'][:5]:
                print(f"  - '{query}'")

        print("=" * 60 + "\n")


# ============================================================================
# SECTION 3: TESTING AND EVALUATION
# ============================================================================

def test_threshold_sensitivity():
    """Test chatbot performance across different threshold values."""
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    test_cases = [
        ("When is the deadline?", "deadlines"),
        ("What's the due date for submission?", "deadlines"),
        ("How do I submit my work?", "submission_format"),
        ("What format should the report be?", "submission_format"),
        ("How is it marked?", "marking_criteria"),
        ("What are the grading criteria?", "marking_criteria"),
        ("Who teaches this course?", "module_leader"),
        ("Who is the lecturer?", "module_leader"),
        ("What is plagiarism?", "plagiarism"),
        ("How many words for report?", "word_count"),
        ("What is LSEPI?", "lsepi"),
        ("How many people in a group?", "group_size"),
        ("Can I submit late?", "late_submission"),
        ("What grade do I need to pass?", "grade_boundaries"),
        ("random unrelated query xyz", None),  # Should not match
    ]

    results = []

    print("\nTESTING THRESHOLD SENSITIVITY")
    print("=" * 80)
    print(f"{'Threshold':<12} {'Correct':<10} {'Incorrect':<12} {'No Response':<14} {'Accuracy':<10}")
    print("-" * 80)

    for threshold in thresholds:
        chatbot = TFIDFChatbot(faq_knowledge_base, threshold=threshold)

        correct = 0
        incorrect = 0
        no_response = 0

        for test_query, expected_intent in test_cases:
            _, score, actual_intent, _ = chatbot.find_best_match(test_query)
            meets_threshold = score >= threshold

            if expected_intent is None:
                # Should NOT match
                if not meets_threshold:
                    correct += 1
                else:
                    incorrect += 1
            else:
                # Should match
                if meets_threshold and actual_intent == expected_intent:
                    correct += 1
                elif meets_threshold and actual_intent != expected_intent:
                    incorrect += 1
                else:
                    no_response += 1

        total = len(test_cases)
        accuracy = (correct / total) * 100

        results.append({
            'threshold': threshold,
            'correct': correct,
            'incorrect': incorrect,
            'no_response': no_response,
            'accuracy': accuracy
        })

        print(f"{threshold:<12.1f} {correct:<10} {incorrect:<12} {no_response:<14} {accuracy:.1f}%")

    print("=" * 80 + "\n")
    return results


def analyze_tfidf_features(chatbot, top_n=10):
    """Analyze most important TF-IDF features."""
    feature_names = chatbot.vectorizer.get_feature_names_out()

    # Get average TF-IDF scores for each term
    tfidf_scores = np.asarray(chatbot.tfidf_matrix.mean(axis=0)).flatten()

    # Get top features
    top_indices = np.argsort(tfidf_scores)[-top_n:][::-1]

    print("\nTOP TF-IDF FEATURES (Most Discriminative Terms)")
    print("=" * 50)
    for idx in top_indices:
        print(f"{feature_names[idx]:<20} {tfidf_scores[idx]:.4f}")
    print("=" * 50 + "\n")


def test_paraphrase_robustness(chatbot):
    """Test how well chatbot handles paraphrased questions."""
    paraphrase_tests = [
        # Original vs paraphrased
        ("When is the deadline?", "What's the last date to submit?"),
        ("How do I submit?", "What's the process for handing in work?"),
        ("Who is the module leader?", "Which professor teaches this?"),
        ("How many words?", "What's the length requirement?"),
        ("What is plagiarism?", "Can you explain academic dishonesty?"),
    ]

    print("\nPARAPHRASE ROBUSTNESS TEST")
    print("=" * 80)
    print(f"{'Original Query':<35} {'Paraphrased Query':<35} {'Match?':<10}")
    print("-" * 80)

    for original, paraphrase in paraphrase_tests:
        _, score1, intent1, _ = chatbot.find_best_match(original)
        _, score2, intent2, _ = chatbot.find_best_match(paraphrase)

        match = "✓" if intent1 == intent2 else "✗"
        print(f"{original:<35} {paraphrase:<35} {match:<10}")
        print(f"  Similarity: {score1:.3f} vs {score2:.3f}")

    print("=" * 80 + "\n")


def visualize_similarity_matrix(chatbot):
    """Create visualization of similarity patterns."""
    # Sample queries from each category
    sample_queries = [
        "When is the deadline?",
        "How do I submit?",
        "Who is the lecturer?",
        "How is it marked?",
        "How many in a group?",
        "What is plagiarism?"
    ]

    # Calculate similarities
    similarity_matrix = []
    for query in sample_queries:
        query_vector = chatbot.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, chatbot.tfidf_matrix)[0]
        # Get max similarity per intent
        intent_sims = []
        for intent in faq_knowledge_base.keys():
            intent_indices = [i for i, name in enumerate(chatbot.intent_names)
                              if name == intent]
            if intent_indices:
                intent_sims.append(max(similarities[intent_indices]))
            else:
                intent_sims.append(0)
        similarity_matrix.append(intent_sims)

    # Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(similarity_matrix,
                xticklabels=list(faq_knowledge_base.keys()),
                yticklabels=[q[:30] + "..." for q in sample_queries],
                annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('TF-IDF Cosine Similarity Heatmap')
    plt.xlabel('FAQ Intent Categories')
    plt.ylabel('Sample User Queries')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('tfidf_similarity_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: tfidf_similarity_heatmap.png")
    plt.close()


# ============================================================================
# SECTION 4: INTERACTIVE CHAT INTERFACE
# ============================================================================

def run_chatbot():
    """Run interactive chatbot session."""
    chatbot = TFIDFChatbot(faq_knowledge_base, threshold=0.3)

    print("\n" + "=" * 70)
    print("  UNIVERSITY FAQ CHATBOT - TF-IDF with Cosine Similarity")
    print("  COMP1827 Introduction to Artificial Intelligence")
    print("=" * 70)
    print("\nI use TF-IDF vectors and cosine similarity to find answers!")
    print("Type 'help' for examples, 'stats' for performance, or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\n\nBot: Goodbye!")
            break

        if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
            print("Bot: Goodbye! Good luck with your coursework!")
            break

        if user_input.lower() == 'help':
            print("\nBot: Here are some example questions:")
            print("  - When is the deadline?")
            print("  - How should I submit my report?")
            print("  - What are the marking criteria?")
            print("  - How many words should the report be?")
            print("  - What is plagiarism?")
            print("  - Who is the module leader?")
            print("  - What is LSEPI?\n")
            continue

        if user_input.lower() == 'stats':
            chatbot.display_stats()
            continue

        if user_input.lower() == 'analyze':
            analyze_tfidf_features(chatbot)
            continue

        # Get response
        response, score, intent, top_matches = chatbot.find_best_match(user_input)

        if response:
            print(f"\nBot: {response}")
            print(f"     [Similarity: {score:.3f}, Intent: {intent}]")

            # Show top 3 matches for transparency
            if len(top_matches) > 1:
                print(f"     Top matches:")
                for i, match in enumerate(top_matches[:3], 1):
                    print(f"       {i}. {match['question'][:50]}... ({match['similarity']:.3f})")
            print()
        else:
            print(f"\nBot: I'm not confident about that question. [Similarity: {score:.3f}]")
            print("     Try asking about: deadlines, submissions, marking, groups, or plagiarism.\n")


# ============================================================================
# SECTION 5: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TF-IDF FAQ CHATBOT - COMPREHENSIVE TESTING")
    print("=" * 70)

    # Initialize chatbot
    chatbot = TFIDFChatbot(faq_knowledge_base, threshold=0.3)

    print(f"\nChatbot initialized with {len(chatbot.questions)} FAQ questions")
    print(f"Vocabulary size: {len(chatbot.vectorizer.get_feature_names_out())} terms")

    # Run tests
    print("\n" + "=" * 70)
    print("TEST 1: THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 70)
    threshold_results = test_threshold_sensitivity()

    print("\n" + "=" * 70)
    print("TEST 2: TF-IDF FEATURE ANALYSIS")
    print("=" * 70)
    analyze_tfidf_features(chatbot)

    print("\n" + "=" * 70)
    print("TEST 3: PARAPHRASE ROBUSTNESS")
    print("=" * 70)
    test_paraphrase_robustness(chatbot)

    print("\n" + "=" * 70)
    print("TEST 4: SIMILARITY HEATMAP GENERATION")
    print("=" * 70)
    visualize_similarity_matrix(chatbot)

    print("\n" + "=" * 70)
    print("STARTING INTERACTIVE CHAT SESSION")
    print("=" * 70)

    # Start interactive session
    run_chatbot()