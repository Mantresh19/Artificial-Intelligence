Rule-Based FAQ Chatbot for Student Support
An intelligent keyword-matching chatbot designed to assist university students with coursework-related queries.

Core Logic

Scoring Engine: Implements a custom matching algorithm that accounts for exact matches, partial word matching, and phrase length.

Optimized Accuracy: Achieved 86.7% accuracy through rigorous testing of confidence thresholds.

Knowledge Base: Stores 18 categorized FAQ entries with priority-based tie-breaking logic.

System Architecture

The system consists of four modules:

Preprocessing: Lowercase conversion and punctuation removal.

Scoring Engine: Calculates match scores based on keyword sets.

Threshold Filter: Rejects matches below a 0.3 confidence score to prevent false positives.

Response Generator: Delivers the highest-priority matched response.

Technologies

Language: Python

Methodology: Rule-based pattern matching
