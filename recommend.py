import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load data
df = pd.read_csv("data/jobs.csv")

# vectorize
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(df['description'])

# user input
user_input = input("Enter your skills: ")

# convert user input
user_vec = vectorizer.transform([user_input])

# similarity
similarity = cosine_similarity(user_vec, vectors)

# best match
index = similarity.argmax()

print("Best Job for you:", df.iloc[index]['title'])