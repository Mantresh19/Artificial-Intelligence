# Rule-Based FAQ Chatbot for University Student Support
# Author: Liladhar Nilesh Patwardhan
# COMP1827 Introduction to Artificial Intelligence

# SECTION 1: FAQ KNOWLEDGE BASE

faq_knowledge_base = {
    # Deadline related queries
    "deadlines": {
        "keywords": ["deadline", "due date", "when due", "submission date", "hand in", "submit by"],
        "response": "The group presentation is on Monday, 1 December 2025. The individual report is due on Wednesday, 3 December 2025 at 5 PM.",
        "category": "assessment",
        "priority": 3
    },
    "late_submission": {
        "keywords": ["late", "extension", "late submission", "missed deadline", "submit late"],
        "response": "Late submissions are subject to university penalties. Check the coursework regulations at http://www2.gre.ac.uk/current-students/regs or apply for Extenuating Circumstances if you have valid reasons.",
        "category": "assessment",
        "priority": 2
    },
    # Module information
    "module_leader": {
        "keywords": ["module leader", "lecturer", "professor", "teacher", "who teaches", "course leader"],
        "response": "The module leader for COMP1827 is Dr. Mohammad M. al-Rifaie.",
        "category": "module_info",
        "priority": 2
    },
    "module_name": {
        "keywords": ["module name", "course name", "what module", "module code", "comp1827"],
        "response": "This module is COMP1827 Introduction to Artificial Intelligence.",
        "category": "module_info",
        "priority": 1
    },
    # Submission requirements
    "submission_format": {
        "keywords": ["submission", "submit", "format", "pdf", "latex", "how to submit", "upload"],
        "response": "You must submit a single PDF document via Moodle. The report should be generated from LaTeX using the provided template. Make sure files are virus-free and not password protected.",
        "category": "submission",
        "priority": 3
    },
    "word_count": {
        "keywords": ["word count", "word limit", "how many words", "length", "how long"],
        "response": "The individual report should be 1,500 words including references.",
        "category": "submission",
        "priority": 2
    },
    # Marking and assessment
    "marking_criteria": {
        "keywords": ["marking", "marks", "grading", "criteria", "how marked", "assessment criteria"],
        "response": "The report is marked on: Understanding of Problem Domain (30%), Development and Implementation (50%), Conclusions and Critical Review (10%), and Writing Quality with referencing (10%).",
        "category": "marking",
        "priority": 3
    },
    "presentation_marks": {
        "keywords": ["presentation", "demo", "present", "group presentation"],
        "response": "The group presentation is worth 50% of total marks. It is assessed on: Understanding of Problem Domain (30%), Development and Implementation (50%), and Conclusions and Critical Review (20%).",
        "category": "marking",
        "priority": 3
    },
    "grade_boundaries": {
        "keywords": ["grade", "first", "2:1", "pass", "fail", "percentage", "boundaries"],
        "response": "Grade boundaries: 80%+ is exceptional, 70-79% shows high standard, 60-69% demonstrates clear awareness, 50-59% is mainly descriptive, 40-49% is superficial, below 40% shows little evidence of criteria.",
        "category": "marking",
        "priority": 2
    },
    # Group work
    "group_size": {
        "keywords": ["group size", "how many people", "team size", "members", "group members"],
        "response": "Students should work in groups of 4 people. All members should contribute equally to the group work.",
        "category": "group_work",
        "priority": 2
    },
    "group_work": {
        "keywords": ["group work", "team work", "collaborate", "group project", "working together"],
        "response": "Group activities include: discussing topics, identifying individual research areas, sharing progress, and preparing the presentation together. The presentation mark is shared equally among group members.",
        "category": "group_work",
        "priority": 2
    },
    # Technical requirements
    "plagiarism": {
        "keywords": ["plagiarism", "copying", "cheating", "academic offence", "reference"],
        "response": "Plagiarism includes copying from web/books without referencing, submitting joint work as individual, or copying other students. All material must be properly referenced. Work is checked for plagiarism.",
        "category": "academic_integrity",
        "priority": 3
    },
    "lsepi": {
        "keywords": ["lsepi", "ethical", "legal", "social", "professional", "ethics"],
        "response": "You must consider Legal, Social, Ethical and Professional Issues (LSEPI) in your work. This includes thinking about how AI systems impact society, privacy concerns, and professional responsibilities.",
        "category": "requirements",
        "priority": 2
    },
    # Contact and help
    "contact": {
        "keywords": ["contact", "email", "help", "support", "question", "ask"],
        "response": "For questions, contact your lecturer via university email, check Moodle announcements, or visit during lab sessions for topic approval.",
        "category": "support",
        "priority": 1
    },
    "moodle": {
        "keywords": ["moodle", "online", "website", "portal", "course page"],
        "response": "All coursework materials, submission links, and announcements are available on the COMP1827 Moodle page.",
        "category": "support",
        "priority": 1
    },
    # Learning outcomes
    "learning_outcomes": {
        "keywords": ["learning outcome", "objectives", "what will i learn", "skills"],
        "response": "By completing this course you will: understand AI methods and their applications, identify knowledge representation models and search algorithms, and be aware of philosophical and ethical issues in AI.",
        "category": "module_info",
        "priority": 1
    },
    # Workload
    "workload": {
        "keywords": ["hours", "time", "how long", "workload", "effort"],
        "response": "This coursework should take approximately 50 hours for an average student who is up-to-date with tutorial work.",
        "category": "module_info",
        "priority": 1
    }
}

# SECTION 2: CHATBOT CONFIGURATION PARAMETERS

# These parameters can be adjusted to tune the chatbot's behaviour
# Minimum score needed to consider a match valid (0.0 to 1.0)
# Lower value = more lenient matching, Higher value = stricter matching
CONFIDENCE_THRESHOLD = 0.1
# Bonus score for exact phrase matches vs partial word matches
EXACT_MATCH_BONUS = 0.5
# Whether to give partial credit for substring matches
ALLOW_PARTIAL_MATCHING = True
# Minimum word length to consider for matching (filters out "a", "is", etc.)
MIN_WORD_LENGTH = 3

# SECTION 3: MATCHING AND SCORING FUNCTIONS

def preprocess_input(text):
    # Convert to lowercase for case-insensitive matching
    text = text.lower().strip()

    # Remove common punctuation that might interfere with matching
    punctuation = ["?", "!", ".", ",", "'", '"']
    for char in punctuation:
        text = text.replace(char, "")
    return text

def calculate_match_score(user_input, keywords):

    # Calculate how well the user input matches a set of keywords.
    # Returns a score between 0 and 1, where 1 is a perfect match.

    if not keywords:
        return 0.0

    total_score = 0.0
    matches_found = 0

    for keyword in keywords:
        keyword = keyword.lower()

        # Check for exact phrase match (best case)
        if keyword in user_input:
            keyword_length_factor = len(keyword.split()) / 3  # normalise by typical max phrase length
            total_score += 1.0 + EXACT_MATCH_BONUS + keyword_length_factor
            matches_found += 1

        # Check for partial matching if enabled
        elif ALLOW_PARTIAL_MATCHING:
            # Split multi-word keywords and check individual words
            keyword_words = keyword.split()
            words_matched = 0

            for word in keyword_words:
                if len(word) >= MIN_WORD_LENGTH and word in user_input:
                    words_matched += 1
            # Give partial credit based on how many words matched
            if words_matched > 0:
                partial_score = words_matched / len(keyword_words) * 0.5
                total_score += partial_score
                matches_found += 1

    # Normalise score based on number of keywords
    # This prevents entries with many keywords from always winning
    if matches_found > 0:
        # Average the scores and cap at 1.0
        final_score = min(total_score / len(keywords), 1.0)
    else:
        final_score = 0.0
    return final_score


def find_best_match(user_input):
    # Clean the input first
    processed_input = preprocess_input(user_input)

    best_match = None
    best_score = 0.0
    best_intent = None

    # Track all matches for analysis
    all_matches = []

    # Go through each FAQ entry and calculate match score
    for intent_name, faq_entry in faq_knowledge_base.items():
        score = calculate_match_score(processed_input, faq_entry["keywords"])

        # Store for analysis
        if score > 0:
            all_matches.append({
                "intent": intent_name,
                "score": score,
                "priority": faq_entry["priority"]
            })

        # Check if this is the best match so far
        # Use priority as tie-breaker when scores are very close
        if score > best_score:
            best_score = score
            best_match = faq_entry["response"]
            best_intent = intent_name
        elif score > 0 and abs(score - best_score) < 0.1:
            # Scores are close, use priority to decide
            if faq_entry["priority"] > faq_knowledge_base[best_intent]["priority"]:
                best_score = score
                best_match = faq_entry["response"]
                best_intent = intent_name

    return best_match, best_score, best_intent, all_matches

# SECTION 4: CHATBOT RESPONSE GENERATION

def get_chatbot_response(user_input):
    # Handle empty input
    if not user_input.strip():
        return "Please type a question about the university or coursework."

    # Find the best matching FAQ entry
    response, confidence, intent, all_matches = find_best_match(user_input)

    # Check if confidence meets our threshold
    if confidence >= CONFIDENCE_THRESHOLD and response:
        return response
    else:
        # No good match found - return helpful fallback message
        return "I'm not sure about that. Try asking about: deadlines, submission format, marking criteria, group work, plagiarism, or contact information."

# SECTION 5: CONVERSATION LOGGING FOR ANALYSIS
# This helps us analyse chatbot performance and identify gaps

conversation_log = []

def log_interaction(user_input, response, confidence, intent):

    interaction = {
        "user_input": user_input,
        "response": response,
        "confidence": confidence,
        "matched_intent": intent,
        "was_successful": confidence >= CONFIDENCE_THRESHOLD
    }
    conversation_log.append(interaction)

def get_performance_stats():
    if not conversation_log:
        return {"message": "No interactions logged yet."}

    total = len(conversation_log)
    successful = sum(1 for log in conversation_log if log["was_successful"])
    failed = total - successful

    # Calculate average confidence for successful matches
    successful_confidences = [log["confidence"] for log in conversation_log if log["was_successful"]]
    avg_confidence = sum(successful_confidences) / len(successful_confidences) if successful_confidences else 0

    # Find which intents are matched most often
    intent_counts = {}
    for log in conversation_log:
        intent = log["matched_intent"]
        if intent:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

    # Find queries that failed to match
    failed_queries = [log["user_input"] for log in conversation_log if not log["was_successful"]]

    return {
        "total_interactions": total,
        "successful_matches": successful,
        "failed_matches": failed,
        "success_rate": round(successful / total * 100, 1),
        "average_confidence": round(avg_confidence, 3),
        "intent_frequency": intent_counts,
        "failed_queries": failed_queries
    }

def display_stats():
    """Print performance statistics in a readable format."""
    stats = get_performance_stats()

    if "message" in stats:
        print(stats["message"])
        return

    print("\n" + "=" * 50)
    print("CHATBOT PERFORMANCE STATISTICS")
    print("=" * 50)
    print(f"Total interactions: {stats['total_interactions']}")
    print(f"Successful matches: {stats['successful_matches']}")
    print(f"Failed matches: {stats['failed_matches']}")
    print(f"Success rate: {stats['success_rate']}%")
    print(f"Average confidence: {stats['average_confidence']}")

    if stats["intent_frequency"]:
        print("\nMost common topics:")
        sorted_intents = sorted(stats["intent_frequency"].items(), key=lambda x: x[1], reverse=True)
        for intent, count in sorted_intents[:5]:
            print(f"  - {intent}: {count} times")

    if stats["failed_queries"]:
        print("\nQueries that failed to match:")
        for query in stats["failed_queries"][:5]:
            print(f"  - '{query}'")

    print("=" * 50 + "\n")

# SECTION 6: MAIN CHAT LOOP

def run_chatbot():
    print("\n" + "=" * 55)
    print("  UNIVERSITY FAQ CHATBOT - Rule-Based System")
    print("  COMP1827 Introduction to Artificial Intelligence")
    print("=" * 55)
    print("\nHello! I can answer questions about your coursework.")
    print("Type 'help' for example questions, 'stats' for performance,")
    print("or 'quit' to exit.\n")

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\n\nBot: Goodbye!")
            break

        # Check for exit commands
        if user_input.lower() in ["quit", "exit", "bye", "q"]:
            print("Bot: Goodbye! Good luck with your coursework!")
            break

        # Check for help command
        if user_input.lower() == "help":
            print("\nBot: Here are some things you can ask me about:")
            print("  - When is the deadline?")
            print("  - How do I submit my work?")
            print("  - What are the marking criteria?")
            print("  - How many words for the report?")
            print("  - What is plagiarism?")
            print("  - Who is the module leader?")
            print("  - What is LSEPI?")
            print("  - How many people in a group?\n")
            continue

        # Check for stats command
        if user_input.lower() == "stats":
            display_stats()
            continue

        # Get response from chatbot
        response, confidence, intent, all_matches = find_best_match(user_input)

        # Use threshold to decide if match is good enough
        if confidence >= CONFIDENCE_THRESHOLD and response:
            final_response = response
        else:
            final_response = "I'm not sure about that. Try asking about: deadlines, submission format, marking criteria, group work, plagiarism, or contact information."
            intent = None

        # Log this interaction for analysis
        log_interaction(user_input, final_response, confidence, intent)

        # Display response with confidence (helpful for testing/debugging)
        print(f"Bot: {final_response}")
        print(f"     [Confidence: {confidence:.2f}, Intent: {intent}]\n")

# Bonus: Function to test multiple thresholds automatically
def test_all_thresholds():
    """
    Test the chatbot with different threshold values.
    This generates data for your report table.
    """
    global CONFIDENCE_THRESHOLD

    thresholds_to_test = [0.1, 0.3, 0.5, 0.7, 0.9]
    print()
    print("TESTING MULTIPLE THRESHOLD VALUES")
    print()
    print(f"{'Threshold':<12} {'Correct':<10} {'Incorrect':<12} {'No Response':<14} {'Accuracy':<10}")

    for threshold in thresholds_to_test:
        CONFIDENCE_THRESHOLD = threshold

        # Run test cases and count results
        correct = 0
        incorrect = 0
        no_response = 0

        test_cases = [
            ("When is the deadline?", "deadlines"),
            ("What is the due date for submission?", "deadlines"),
            ("How do I submit my work?", "submission_format"),
            ("What format should my report be in?", "submission_format"),
            ("How is the report marked?", "marking_criteria"),
            ("What are the marking criteria?", "marking_criteria"),
            ("Who is the module leader?", "module_leader"),
            ("Who teaches this course?", "module_leader"),
            ("What is plagiarism?", "plagiarism"),
            ("How many words?", "word_count"),
            ("What is LSEPI?", "lsepi"),
            ("How many people in a group?", "group_size"),
            ("Can I submit late?", "late_submission"),
            ("What grade do I need to pass?", "grade_boundaries"),
            ("random nonsense xyz", None),
        ]

        total_tests = len(test_cases)

        for test_input, expected_intent in test_cases:
            response, confidence, actual_intent, _ = find_best_match(test_input)
            meets_threshold = confidence >= CONFIDENCE_THRESHOLD

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

        # Calculate accuracy
        accuracy = (correct / total_tests) * 100

        print(f"{threshold:<12} {correct:<10} {incorrect:<12} {no_response:<14} {accuracy:.1f}%")

    # Reset to default
    CONFIDENCE_THRESHOLD = 0.3

# Run it
if __name__ == "__main__":
    test_all_thresholds()
    # run_chatbot()