
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and clean the data (already done earlier)
df = pd.read_csv("/home/stemland/Downloads/stream_of_thoughts.csv")
columns_to_use = ['Raw', 'Text', 'Text.1', 'Unnamed: 4', 'Unnamed: 5']
df['Raw Text'] = df[columns_to_use].astype(str).agg(' '.join, axis=1)
df['Raw Text'] = df['Raw Text'].str.strip()
df = df[['Raw Text']].dropna()
df = df[df['Raw Text'].str.lower() != 'nan']
df = df.sample(frac=1).reset_index(drop=True)

# Split data
train_data = df[:100]
remaining_data = df[100:]
test_data = remaining_data.sample(n=10, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['Raw Text'])
X_test = vectorizer.transform(test_data['Raw Text'])

# After calculating cosine similarities and printing each test thought's similarity
similarities = cosine_similarity(X_test, X_train)
known = 0
for i, row in enumerate(similarities):
    max_score = row.max()
    print(f"\nTest Thought {i+1}: {test_data['Raw Text'].iloc[i]}")
    print(f"Most similar training thought: {train_data['Raw Text'].iloc[row.argmax()]}")
    print(f"Similarity Score: {max_score:.2f}")
    if max_score > 0.7:
        print("✅ This is a known or similar thought.")
        known += 1
    else:
        print("❌ This might be a new or different thought.")

# Add the count summary here
print(f"\n Known thoughts: {known}")
print(f"New or different thoughts: {len(test_data) - known}")
