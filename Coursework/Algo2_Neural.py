"""
Neural Network FAQ Chatbot for University Student Support
Algorithm 4: Neural Network with TensorFlow/Keras
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random


class NeuralNetworkChatbot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_sequence_length = 20
        self.vocab_size = 1000

        # Training data - questions and their categories
        self.training_data = [
            # Deadline questions
            ("what is the deadline", "deadline"),
            ("when is coursework due", "deadline"),
            ("submission date", "deadline"),
            ("when to submit assignment", "deadline"),
            ("final deadline", "deadline"),
            ("last date for submission", "deadline"),

            # Assignment questions
            ("what are the assignments", "assignment"),
            ("coursework requirements", "assignment"),
            ("what to submit", "assignment"),
            ("project details", "assignment"),
            ("report requirements", "assignment"),

            # Lecture questions
            ("when are lectures", "lecture"),
            ("class schedule", "lecture"),
            ("timetable", "lecture"),
            ("lecture times", "lecture"),
            ("when do classes happen", "lecture"),

            # Professor questions
            ("professor contact", "professor"),
            ("teacher email", "professor"),
            ("instructor details", "professor"),
            ("who is the professor", "professor"),

            # Library questions
            ("library hours", "library"),
            ("when is library open", "library"),
            ("study space availability", "library"),
            ("library timing", "library"),

            # Software questions
            ("required software", "software"),
            ("what tools needed", "software"),
            ("install python", "software"),
            ("latex editor", "software")
        ]

        # Responses for each category
        self.responses = {
            "deadline": "The coursework deadline is December 3rd, 2025 at 5:00 PM.",
            "assignment": "You need to submit a 1500-word individual report and a group presentation video. Both are worth 50% of the final grade.",
            "lecture": "Lectures are held on Mondays (2:00-4:00 PM) and Wednesdays (10:00-12:00 PM) in Room 301.",
            "professor": "The course professor is Dr. Smith. Email: d.smith@university.edu, Office: Building A, Room 205.",
            "library": "The university library is open Monday-Friday 8:00 AM-10:00 PM, Weekends 9:00 AM-8:00 PM.",
            "software": "Required software: Python 3.8+, LaTeX editor, VS Code. All available on the university software portal.",
            "unknown": "I'm not sure about that. I can help with deadlines, assignments, lectures, professor contacts, library info, or software requirements."
        }

    def preprocess_data(self):
        """Prepare training data for neural network"""
        questions = [data[0] for data in self.training_data]
        labels = [data[1] for data in self.training_data]

        # Create tokenizer and convert text to sequences
        self.tokenizer = keras.preprocessing.text.Tokenizer(
            num_words=self.vocab_size,
            oov_token="<OOV>"
        )
        self.tokenizer.fit_on_texts(questions)

        # Convert text to sequences
        sequences = self.tokenizer.texts_to_sequences(questions)
        X = keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )

        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)

        return X, y

    def build_model(self):
        """Build and compile the neural network model"""
        model = keras.Sequential([
            keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=16,
                input_length=self.max_sequence_length
            ),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(12, activation='relu'),
            keras.layers.Dense(len(self.label_encoder.classes_), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_model(self, epochs=100):
        """Train the neural network model"""
        print("🔄 Preparing training data...")
        X, y = self.preprocess_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("🔄 Building neural network model...")
        self.model = self.build_model()

        print("🔄 Training neural network...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=0
        )

        # Evaluate model
        train_accuracy = self.model.evaluate(X_train, y_train, verbose=0)[1]
        test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)[1]

        print(f"✅ Training completed!")
        print(f"📊 Training Accuracy: {train_accuracy:.2%}")
        print(f"📊 Test Accuracy: {test_accuracy:.2%}")

        return history

    def predict_intent(self, user_input):
        """Predict the intent of user input using neural network"""
        if self.model is None:
            return "unknown"

        # Preprocess input
        sequence = self.tokenizer.texts_to_sequences([user_input])
        padded_sequence = keras.preprocessing.sequence.pad_sequences(
            sequence,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )

        # Make prediction
        prediction = self.model.predict(padded_sequence, verbose=0)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        # Only return prediction if confidence is high enough
        if confidence > 0.6:
            predicted_class = self.label_encoder.inverse_transform([predicted_class_index])[0]
            return predicted_class, confidence
        else:
            return "unknown", confidence

    def get_response(self, user_input):
        """Get response based on neural network prediction"""
        intent, confidence = self.predict_intent(user_input)
        response = self.responses.get(intent, self.responses["unknown"])
        return response, intent, confidence

    def run_chatbot(self):
        """Main function to run the neural network chatbot"""
        print("🧠 Neural Network FAQ Chatbot")
        print("=" * 50)
        print("Training the neural network on university FAQs...")

        # Train the model
        self.train_model(epochs=80)

        print("\n" + "=" * 50)
        print("Chatbot is ready! Ask me anything about:")
        print("- Deadlines and due dates")
        print("- Assignment requirements")
        print("- Lecture schedules")
        print("- Professor contacts")
        print("- Library information")
        print("- Software requirements")
        print("Type 'exit' to quit")
        print("=" * 50)

        while True:
            user_input = input("\nYou: ").strip().lower()

            if user_input in ['exit', 'quit', 'bye']:
                print("🤖 Thank you for chatting! Goodbye!")
                break

            if user_input:
                response, intent, confidence = self.get_response(user_input)
                print(f"🤖 Bot: {response}")
                print(f"   [Predicted: {intent} | Confidence: {confidence:.1%}]")

    def display_model_summary(self):
        """Display neural network architecture"""
        if self.model:
            print("\n" + "=" * 50)
            print("NEURAL NETWORK ARCHITECTURE")
            print("=" * 50)
            self.model.summary()


# Test the neural network with sample queries
def test_neural_network():
    """Test the neural network with various inputs"""
    chatbot = NeuralNetworkChatbot()

    print("🧪 Testing Neural Network Chatbot")
    print("=" * 40)

    test_queries = [
        "what is the deadline for coursework",
        "when to submit assignment",
        "lecture schedule",
        "professor email address",
        "library opening times",
        "what software do I need",
        "when are classes",
        "tell me about the project"
    ]

    # Train model first
    chatbot.train_model(epochs=50)

    for query in test_queries:
        response, intent, confidence = chatbot.get_response(query)
        print(f"Query: '{query}'")
        print(f"Response: {response}")
        print(f"Intent: {intent} | Confidence: {confidence:.1%}")
        print("-" * 40)


# Main execution
if __name__ == "__main__":
    # Choose between testing or interactive mode
    print("Neural Network Chatbot - Choose Mode:")
    print("1. Run tests")
    print("2. Interactive chatbot")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        test_neural_network()
    else:
        chatbot = NeuralNetworkChatbot()
        chatbot.run_chatbot()
        chatbot.display_model_summary()