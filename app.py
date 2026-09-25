import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Fake News Detection", page_icon="📰")

st.title("📰 Fake News Detection")
st.write("Enter a news article below to check whether it is likely to be Fake or Real.")

fake = pd.read_csv("dataset/fake.csv")
true = pd.read_csv("dataset/true.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)

if "title" in data.columns and "text" in data.columns:
    data["content"] = data["title"].fillna("") + " " + data["text"].fillna("")
elif "text" in data.columns:
    data["content"] = data["text"].fillna("")
elif "title" in data.columns:
    data["content"] = data["title"].fillna("")
else:
    st.error("Dataset does not contain title or text columns.")
    st.stop()

data = data[["content", "label"]].dropna()

X = data["content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

st.info(f"Model Accuracy: {accuracy * 100:.2f}%")

news = st.text_area(
    "Enter news article:",
    height=200,
    placeholder="Paste the news article here..."
)

if st.button("Check News"):
    if news.strip():
        prediction = model.predict([news])[0]
        probability = model.predict_proba([news])[0].max()

        if prediction == 0:
            st.error(f"🚨 FAKE NEWS\n\nConfidence: {probability * 100:.2f}%")
        else:
            st.success(f"✅ REAL NEWS\n\nConfidence: {probability * 100:.2f}%")
    else:
        st.warning("Please enter some news first.")