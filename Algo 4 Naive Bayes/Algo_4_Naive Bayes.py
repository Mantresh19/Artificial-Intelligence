"""
my naive bayes classifier for the faq chatbot
this is my part of the group project
"""

import math
from collections import defaultdict


# intent classifier class
class NBClassifier:

    def __init__(self):
        self.categories = []
        self.words = set()
        self.cat_prob = {}
        self.word_prob = {}
        self.word_count_total = {}
        self.ready = False

    def clean(self, txt):
        # preprocessing step
        txt = txt.lower()
        # remove punctuation
        for c in ".,!?;:'\"":
            txt = txt.replace(c, "")
        return txt.split()

    def fit(self, X, y):
        # training function

        if len(X) != len(y):
            print("error: X and y lengths dont match!")
            return

        self.categories = list(set(y))
        total = len(y)

        # calculate P(category)
        cat_counts = {}
        for cat in self.categories:
            cat_counts[cat] = 0

        for label in y:
            cat_counts[label] = cat_counts[label] + 1

        for cat in self.categories:
            self.cat_prob[cat] = cat_counts[cat] / total

        # count words
        word_freq = {}
        for cat in self.categories:
            word_freq[cat] = defaultdict(int)

        for text, label in zip(X, y):
            tokens = self.clean(text)
            for token in tokens:
                self.words.add(token)
                word_freq[label][token] = word_freq[label][token] + 1

        # total words per category
        for cat in self.categories:
            count = 0
            for word in word_freq[cat]:
                count = count + word_freq[cat][word]
            self.word_count_total[cat] = count

        # calculate P(word|category) - with smoothing
        V = len(self.words)

        for cat in self.categories:
            self.word_prob[cat] = {}
            for word in self.words:
                # smoothing to avoid zero probabilities
                num = word_freq[cat][word] + 1
                denom = self.word_count_total[cat] + V
                self.word_prob[cat][word] = num / denom

        self.ready = True
        print("trained on", len(X), "examples")

    def predict(self, text):
        # make prediction

        if not self.ready:
            return None, {}

        tokens = self.clean(text)
        scores = {}

        for cat in self.categories:
            # use log to prevent underflow
            s = math.log(self.cat_prob[cat])

            for token in tokens:
                if token in self.words:
                    s = s + math.log(self.word_prob[cat][token])

            scores[cat] = s

        # find best category
        best = None
        best_score = float('-inf')
        for cat in scores:
            if scores[cat] > best_score:
                best_score = scores[cat]
                best = cat

        # calculate confidence scores
        max_score = max(scores.values())
        temp = {}
        for cat in scores:
            temp[cat] = math.exp(scores[cat] - max_score)

        sum_temp = sum(temp.values())
        confidence = {}
        for cat in temp:
            confidence[cat] = temp[cat] / sum_temp

        return best, confidence


# my training data
X_train = [
    "how do i apply",
    "what do i need to apply to uni",
    "when is the deadline",
    "application process",
    "admission requirements",
    "can i still apply",
    "what documents for application",

    "how much does it cost",
    "tuition fees",
    "are there scholarships",
    "can i get financial aid",
    "payment deadlines",
    "how expensive is it",
    "do you have payment plans",

    "what courses do you offer",
    "do you have computer science",
    "what can i study",
    "tell me about engineering",
    "course options",
    "what subjects are available",

    "where is the campus",
    "how do i get there",
    "is there accommodation",
    "do you have a library",
    "what facilities",
    "campus location",

    "how can i contact you",
    "what is your email",
    "phone number",
    "how to reach admissions",
    "can i visit",
    "where is your office"
]

y_train = [
    "admissions", "admissions", "admissions", "admissions", "admissions", "admissions", "admissions",
    "fees", "fees", "fees", "fees", "fees", "fees", "fees",
    "courses", "courses", "courses", "courses", "courses", "courses",
    "campus", "campus", "campus", "campus", "campus", "campus",
    "contact", "contact", "contact", "contact", "contact", "contact"
]

# the responses for each category
responses_dict = {
    "admissions": "You can apply online at our website. You need transcripts, personal statement, and a reference letter. Application deadline is January 15th for most courses.",
    "fees": "Tuition fees are £9,250 per year for UK/EU students and £15,000-£20,000 for international students. We have scholarships available - check our funding page.",
    "courses": "We offer a wide range of courses including Computer Science, Engineering, Business, Natural Sciences, and Humanities. Visit our course catalog for details.",
    "campus": "Our main campus is located in the city center with excellent transport links. We have a 24/7 library, modern sports facilities, and on-campus accommodation.",
    "contact": "You can reach us at info@university.ac.uk or call 01234 567890. Our offices are open Monday to Friday, 9am to 5pm. You're also welcome to visit us!"
}


# test function
def test():
    print("\nTesting the classifier...\n")

    clf = NBClassifier()
    clf.fit(X_train, y_train)

    test_questions = [
        "what's the deadline?",
        "how much do i have to pay?",
        "do you teach AI?",
        "where is the university?",
        "i want to get in touch"
    ]

    for q in test_questions:
        cat, conf = clf.predict(q)
        print("Q:", q)
        print("Category:", cat, f"({conf[cat] * 100:.0f}% confident)")
        print("Response:", responses_dict[cat][:60] + "...")
        print()


# interactive chat
def chat():
    clf = NBClassifier()
    clf.fit(X_train, y_train)

    print("\n" + "=" * 50)
    print("University FAQ Bot - Type 'quit' to exit")
    print("=" * 50 + "\n")

    while True:
        q = input("Ask me anything: ")

        if q.lower() in ['quit', 'exit', 'q']:
            print("\nBye!")
            break

        if not q:
            continue

        cat, conf = clf.predict(q)
        print(f"\n[{cat}] {responses_dict[cat]}\n")


if __name__ == "__main__":
    test()

    # uncomment to run interactive mode
    chat()