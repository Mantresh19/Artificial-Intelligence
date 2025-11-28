import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Sample FAQ data - expanded training dataset
questions = [
    # Deadline intent
    "when is the assignment deadline",
    "assignment deadline",
    "when is it due",
    "what is the due date",
    "deadline for submission",
    "when should i submit",
    "submission deadline date",
    "when must i finish",
    # Lecture intent
    "what time is the lecture",
    "lecture time",
    "when is class",
    "what time is the class",
    "lecture schedule",
    "when do we meet",
    "class time",
    "lecture timing",
    # Submission intent
    "how do i submit my work",
    "submit work",
    "how to submit",
    "where do i upload",
    "submission process",
    "how to upload assignment",
    "submit assignment",
    "upload work",
    # Exam intent
    "where is the exam",
    "exam location",
    "when is the exam",
    "exam details",
    "exam time and place",
    "exam schedule",
    "where do i sit exam",
    "exam venue",
    # Grades intent
    "what is my grade",
    "check grades",
    "my marks",
    "grade results",
    "when will i get grades",
    "grades release",
    "my score",
    "feedback marks",
    # Contact intent
    "how do i contact my tutor",
    "contact tutor",
    "tutor email",
    "reach out to tutor",
    "tutor office hours",
    "how to contact professor",
    "tutor phone",
    "get in touch with tutor",
    # Schedule intent
    "class schedule",
    "lecture schedule",
    "timetable",
    "course timetable",
    "when are classes",
    "schedule of classes",
    "next class time",
    "course schedule",
    # Library intent
    "how do i access the library",
    "library access",
    "library hours",
    "when is library open",
    "library opening times",
    "library location",
    "access library",
    "use library resources"
]

# Intent categories and corresponding answers
intents = ["deadline", "deadline", "deadline", "deadline", "deadline", "deadline", "deadline", "deadline",
           "lecture", "lecture", "lecture", "lecture", "lecture", "lecture", "lecture", "lecture",
           "submission", "submission", "submission", "submission", "submission", "submission", "submission",
           "submission",
           "exam", "exam", "exam", "exam", "exam", "exam", "exam", "exam",
           "grades", "grades", "grades", "grades", "grades", "grades", "grades", "grades",
           "contact", "contact", "contact", "contact", "contact", "contact", "contact", "contact",
           "schedule", "schedule", "schedule", "schedule", "schedule", "schedule", "schedule", "schedule",
           "library", "library", "library", "library", "library", "library", "library", "library"]

answers = {
    "deadline": "The assignment deadline is Monday, December 1st, 2025 at 5pm. Please submit your work on Moodle.",
    "lecture": "Lectures are held on Wednesdays and Fridays at 10am in Building 1, Room 101.",
    "submission": "You can submit your work through the Moodle page for your course. Click on the assignment link and upload your PDF file.",
    "exam": "The exam will be held on December 15th, 2025 in the Main Examination Hall from 2pm to 4pm.",
    "grades": "Your grades will be released on the course Moodle page within 2 weeks of submission.",
    "contact": "You can contact your tutor during office hours (Tuesdays 2-4pm) or email them at tutor@gre.ac.uk",
    "schedule": "Check the course timetable on Moodle for the full lecture schedule.",
    "library": "The university library is open Monday-Friday 8am-6pm and Saturday 10am-4pm. Access it via your student card."
}

# Tokenizer for text preprocessing
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(questions)
X_train_seq = tokenizer.texts_to_sequences(questions)
X_train_pad = pad_sequences(X_train_seq, maxlen=10, padding='post')

# Convert labels to numeric
intent_list = intents
label_map = {label: idx for idx, label in enumerate(set(intent_list))}
y_train_encoded = np.array([label_map[label] for label in intent_list])

print("Training neural network model...")

# Build neural network model with embedding layer
model = Sequential([
    Input(shape=(10,)),
    Embedding(100, 16),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(len(set(intent_list)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train_pad, y_train_encoded, epochs=50, verbose=0, batch_size=2)
train_accuracy = model.evaluate(X_train_pad, y_train_encoded, verbose=0)[1]
print(f"Training complete!")
print(f"Model accuracy on training data: {train_accuracy:.2%}\n")

# Interactive chatbot
reverse_label_map = {v: k for k, v in label_map.items()}

print("=" * 70)
print("FAQ CHATBOT - Neural Network Based Intent Classification")
print("=" * 70)
print("Example questions:")
print("- When is the deadline?")
print("- What time is the lecture?")
print("- How do I submit my work?")
print("- Where is the exam?")
print("- What is my grade?")
print("- How do I contact my tutor?")
print("- When is the next lecture?")
print("- How do I access the library?")
print("Type 'quit' to exit")
print("=" * 70 + "\n")

while True:
    user_question = input("You: ").strip()

    if user_question.lower() == 'quit':
        print("Goodbye!")
        break

    if not user_question:
        print("Please enter a question.\n")
        continue

    # Process user input
    user_seq = tokenizer.texts_to_sequences([user_question])
    user_pad = pad_sequences(user_seq, maxlen=10, padding='post')

    # Make prediction
    prediction = model.predict(user_pad, verbose=0)
    confidence = np.max(prediction[0])
    intent_idx = np.argmax(prediction[0])
    intent = reverse_label_map[intent_idx]

    print(f"Chatbot: {answers[intent]}")
    print(f"(Intent: {intent}, Confidence: {confidence:.2%})\n")