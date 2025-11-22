# naive_bayes_chatbot.py
# Algorithm 2: Naive Bayes Classifier for Intent Recognition
# Author: Liladhar Nilesh Patwardhan

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Training data (examples of possible student questions)
training_sentences = [
    # Deadlines
    "When is the report due?",
    "What is the submission deadline?",
    "When do we need to submit the coursework?",
    "What is the last date for presentation?",

    # Module leader
    "Who is the course leader?",
    "What is the name of the module leader?",
    "Who teaches this subject?",

    # Submission details
    "How should I submit my report?",
    "Where do we upload our assignment?",
    "Is submission through Moodle?",

    # Marking scheme
    "How are we graded?",
    "What is the marking scheme?",
    "How much marks for report or presentation?",

    # LSEPI
    "What is LSEPI?",
    "What are the ethical issues?",
    "Do we need to mention professional issues?",
]

# Step 2: Labels for each sentence (intent categories)
labels = [
    "deadline", "deadline", "deadline", "deadline",
    "module_leader", "module_leader", "module_leader",
    "submission", "submission", "submission",
    "marking", "marking", "marking",
    "lsepi", "lsepi", "lsepi"
]

# Step 3: Convert text into word frequency vectors
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_sentences)

# Step 4: Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, labels)

# Step 5: Dictionary of responses
responses = {
    "deadline": "The presentation is on 1 Dec 2025 and the report is due on 3 Dec 2025 at 5 PM.",
    "module_leader": "The module leader is Dr. Mohammad M. al-Rifaie.",
    "submission": "Submit a single PDF generated from LaTeX via Moodle.",
    "marking": "Report: 30% understanding, 50% implementation, 10% conclusion, 10% writing.",
    "lsepi": "LSEPI covers legal, social, ethical, and professional issues of AI systems.",
    "unknown": "Sorry, I’m not sure about that. Try asking about deadlines or submission."
}

# Step 6: Function to predict intent and give response
def get_response(user_input):
    X_test = vectorizer.transform([user_input])
    predicted_intent = classifier.predict(X_test)[0]
    return responses.get(predicted_intent, responses["unknown"])

# Step 7: Terminal chat loop
print("University FAQ Chatbot (Naive Bayes Version)")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Bot: Goodbye!")
        break
    answer = get_response(user_input)
    print("Bot:", answer)