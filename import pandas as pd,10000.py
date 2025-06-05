import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV file
df = pd.read_csv("C:\\Users\\C3STREAMLAND\\Downloads\\stream_of_thoughts.csv")

# Clean and filter Raw Text column
df = df[['Raw Text']].dropna()

df = df[df['Raw Text'].str.lower() != 'nan']
df = df.sample(frac=1).reset_index(drop=True)

# Split data
train_data = df[:100]
remaining_data = df[100:]
test_data = remaining_data.sample(n=10, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['Raw Text'])
X_test = vectorizer.transform(test_data['Raw Text'])

# Cosine similarity
similarities = cosine_similarity(X_test, X_train)
known = 0
for i, row in enumerate(similarities):
    max_score = row.max()
    print(f"\nTest Thought {i+1}: {test_data['Raw Text'].iloc[i]}")
    print(f"Most similar training thought: {train_data['Raw Text'].iloc[row.argmax()]}")
    print(f"Similarity Score: {max_score:.2f}")
    if max_score > 0.7:
        print(" This is a known or similar thought.")
        known += 1
    else:
        print(" This might be a new or different thought.")

# Summary
print(f"\nKnown thoughts: {known}")
print(f"New or different thoughts: {len(test_data) - known}")
