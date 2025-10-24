# rule_based_chatbot.py
# Simple Rule-Based FAQ Chatbot for University Student Support
# Author: Liladhar Nilesh Patwardhan

# Step 1: Create a small FAQ knowledge base (patterns → responses)
faq_data = {
    "deadline": "The presentation is on 1 Dec 2025 and the report is due on 3 Dec 2025 at 5 PM.",
    "module leader": "The module leader is Dr. Mohammad M. al-Rifaie.",
    "submission": "You must submit a single PDF generated from LaTeX via Moodle.",
    "marking": "Report: 30% problem understanding, 50% implementation, 10% conclusion, 10% writing.",
    "lsepi": "Consider legal, social, ethical, and professional issues (LSEPI) when designing AI systems.",
    "contact": "Please contact your lecturer via university email or check Moodle announcements."
}

# Step 2: Function to find keyword matches
def chatbot_reply(user_input):
    user_input = user_input.lower()
    for keyword, response in faq_data.items():
        if keyword in user_input:
            return response
    return "Sorry, I don't have information on that. Try asking about deadlines, submission, or marking."

# Step 3: Main loop (terminal interaction)
print("University FAQ Chatbot (Rule-Based System)")
print("Type 'quit' to exit.\n")
print("Type deadline, module leader, submission, marking, lsepi, or contact.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Bot: Goodbye!")
        break
    reply = chatbot_reply(user_input)
    print("Bot:", reply)
