import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# 2. Preprocess the text
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = text.strip()
    return text

df['message'] = df['message'].apply(preprocess)

# 3. Convert labels to binary
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

# 5. Vectorize the text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Evaluate the model
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Predict new messages
def predict_message(msg):
    msg = preprocess(msg)
    vec = vectorizer.transform([msg])
    prediction = model.predict(vec)[0]
    return "Spam" if prediction == 1 else "Ham"

# Example usage
print(predict_message("Congratulations! You've won a free ticket to Bahamas. Call now!"))
print(predict_message("Hey, are we still meeting for lunch today?"))
