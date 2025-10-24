import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

class NeuralNetworkChatbot:
    def __init__(self):
        self.model = None
        self.tokenizer = Tokenizer()
        self.label_encoder = LabelEncoder()
        self.max_sequence_length = 20

        # Training data - questions and their categories
        self.training_data = [
            # Deadlines
            {"question": "when is the deadline", "category": "deadline"},
            {"question": "submission date", "category": "deadline"},
            {"question": "when to submit coursework", "category": "deadline"},
            {"question": "final due date", "category": "deadline"},
            {"question": "last date for submission", "category": "deadline"},

            # Assignments
            {"question": "tell me about assignments", "category": "assignment"},
            {"question": "coursework details", "category": "assignment"},
            {"question": "what are the projects", "category": "assignment"},
            {"question": "homework requirements", "category": "assignment"},
            {"question": "assignment submission", "category": "assignment"},

            # Lectures
            {"question": "lecture schedule", "category": "lecture"},
            {"question": "class timetable", "category": "lecture"},
            {"question": "when are classes", "category": "lecture"},
            {"question": "teaching hours", "category": "lecture"},
            {"question": "course schedule", "category": "lecture"},

            # Modules
            {"question": "available modules", "category": "module"},
            {"question": "course list", "category": "module"},
            {"question": "what courses can I take", "category": "module"},
            {"question": "subject options", "category": "module"},
            {"question": "module selection", "category": "module"},

            # Admissions
            {"question": "how to apply", "category": "admission"},
            {"question": "admission process", "category": "admission"},
            {"question": "enrollment procedure", "category": "admission"},
            {"question": "application deadline", "category": "admission"},
            {"question": "admission requirements", "category": "admission"}
        ]

        # Responses for each category
        self.responses = {
            "deadline": [
                "The coursework deadline is December 3rd, 2025.",
                "All submissions are due by December 3rd at 5 PM.",
                "Final deadline: December 3rd, 2025."
            ],
            "assignment": [
                "Assignments include a group project and individual report.",
                "You'll have programming assignments and written coursework.",
                "Check Moodle for specific assignment details and requirements."
            ],
            "lecture": [
                "Lectures are on Monday 2 PM and Wednesday 10 AM.",
                "Class schedule: Mon 2-4 PM, Wed 10-12 PM in Room 301.",
                "You can find the complete timetable on the university app."
            ],
            "module": [
                "Available modules: AI, Programming, Databases, and Web Development.",
                "Core modules include COMP1827 (AI) and COMP1600 (Programming).",
                "You can choose from various computing and mathematics modules."
            ],
            "admission": [
                "Visit university.edu/admissions for application details.",
                "The application deadline is January 31st for next semester.",
                "Contact admissions@university.edu for specific requirements."
            ]
        }

    def prepare_data(self):
        """Prepare training data for the neural network"""
        questions = [item["question"] for item in self.training_data]
        categories = [item["category"] for item in self.training_data]

        # Tokenize text
        self.tokenizer.fit_on_texts(questions)
        sequences = self.tokenizer.texts_to_sequences(questions)
        X = pad_sequences(sequences, maxlen=self.max_sequence_length)

        # Encode labels
        y = self.label_encoder.fit_transform(categories)

        return X, y

    def build_model(self, vocab_size, num_classes):
        """Build a simple neural network model"""
        model = Sequential([
            # Embedding layer for text processing
            Embedding(input_dim=vocab_size + 1, output_dim=64, input_length=self.max_sequence_length),

            # LSTM layer for sequence understanding
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),

            # Dense layers for classification
            Dense(32, activation='relu'),
            Dropout(0.3),

            # Output layer
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, epochs=100):
        """Train the neural network"""
        print("Preparing training data...")
        X, y = self.prepare_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build model
        vocab_size = len(self.tokenizer.word_index)
        num_classes = len(self.label_encoder.classes_)

        print(f"Vocabulary size: {vocab_size}")
        print(f"Number of classes: {num_classes}")

        self.model = self.build_model(vocab_size, num_classes)

        print("Training neural network...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=8,
            validation_data=(X_test, y_test),
            verbose=1
        )

        print("Training completed!")
        return history

    def predict_category(self, question):
        """Predict the category of a question"""
        if self.model is None:
            return "Model not trained yet. Please train the model first."

        # Preprocess the question
        sequence = self.tokenizer.texts_to_sequences([question])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length)

        # Make prediction
        prediction = self.model.predict(padded_sequence, verbose=0)
        predicted_class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        # Get category name
        predicted_category = self.label_encoder.inverse_transform([predicted_class_index])[0]

        return predicted_category, confidence

    def get_response(self, question):
        """Get response for a question"""
        predicted_category, confidence = self.predict_category(question)

        if confidence > 0.6:  # Confidence threshold
            responses = self.responses.get(predicted_category, ["I'm not sure about that."])
            return random.choice(responses), predicted_category, confidence
        else:
            return "I'm not sure I understand. Can you rephrase your question?", "unknown", confidence

    def add_training_example(self, question, category):
        """Add new training example to improve the model"""
        self.training_data.append({"question": question, "category": category})
        print(f"Added training example: '{question}' -> '{category}'")

# Demonstration and testing
def demonstrate_neural_chatbot():
    # Create and train the chatbot
    chatbot = NeuralNetworkChatbot()

    print("=== Neural Network Chatbot Demonstration ===")
    print("Training the model... This may take a few minutes.")

    # Train with fewer epochs for demonstration (use more for better accuracy)
    chatbot.train(epochs=50)

    print("\n=== Testing the Chatbot ===")

    test_questions = [
        "when is the submission deadline",
        "tell me about class schedule",
        "what assignments are required",
        "how to apply for courses",
        "available modules this semester",
        "when should I submit my work",
        "lecture times and dates"
    ]

    for question in test_questions:
        response, category, confidence = chatbot.get_response(question)
        print(f"\nYou: {question}")
        print(f"Bot: {response}")
        print(f"Category: {category}, Confidence: {confidence:.2f}")

    # Interactive mode
    print("\n=== Interactive Mode ===")
    print("Type 'quit' to exit, 'retrain' to train with more data")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'retrain':
            print("Retraining with more epochs...")
            chatbot.train(epochs=100)
        else:
            response, category, confidence = chatbot.get_response(user_input)
            print(f"Bot: {response}")
            print(f"(Category: {category}, Confidence: {confidence:.2f})")

if __name__ == "__main__":
    demonstrate_neural_chatbot()