from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



emails = [
    "Win money now",
    "Claim your free prize",
    "Congratulations you won lottery",
    "Free iphone offer",
    "Click this link to claim reward",
    "Limited offer click this link",
    "Click this link to win cash",
    "Earn money quickly click here",
    "Exclusive offer just for you",

    "Meeting at 5pm",
    "Let's have lunch",
    "Project discussion tomorrow",
    "Are we meeting today",
    "Please review the document",
    "Let's schedule a meeting",
    "See you tomorrow",
    "Lunch tomorrow?",
    "Call me when you are free",
    "Team meeting today",
    "Send the report today"
]

labels = [
    "spam","spam","spam","spam","spam","spam","spam","spam","spam",
    "not spam","not spam","not spam","not spam","not spam",
    "not spam","not spam","not spam","not spam","not spam","not spam"
]



vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(emails)


X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)


model = MultinomialNB()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")



while True:
    user_email = input("\nEnter an email message: ")

    if user_email.lower() == "exit":
        print("Program stopped.")
        break

    email_vector = vectorizer.transform([user_email])
    prediction = model.predict(email_vector)

    if prediction[0] == "spam":
        print("⚠️ This email looks like SPAM")
    else:
        print("✅ This email looks safe (NOT SPAM)")