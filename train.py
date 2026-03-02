import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# load REAL data
df = pd.read_csv("data/real_jobs.csv")


# 🔥 STEP: Auto labeling (YAHI ADD KARNA THA)
def assign_label(text):
    text = text.lower()
    if "machine learning" in text or "ai" in text:
        return "ML"
    elif "web" in text or "react" in text:
        return "Web"
    elif "data" in text or "analyst" in text:
        return "Data"
    else:
        return "Other"

# apply labeling
df['label'] = df['title'].apply(assign_label)


# input and output
X = df['title']
y = df['label']

# vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# model
model = LogisticRegression()
model.fit(X_vec, y)

# test
sample = ["Python machine learning"]
sample_vec = vectorizer.transform(sample)

print("Prediction:", model.predict(sample_vec))