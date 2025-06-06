# Thousand_stream

#  Stream of Thoughts Similarity Checker 

This project checks how similar a new thought is to known thoughts using text analysis techniques like TF-IDF and cosine similarity.

---

##  1. Loading the Data
- The program loads a CSV file that contains a column called **"Raw Text"**.
- It removes any rows that are empty or have invalid values like `'nan'`.
- Then, it shuffles the data randomly so that the selection is unbiased.

---

##  2. Splitting the Data
- The **first 100 thoughts** are selected as the **training data**.
- From the remaining data, **10 random thoughts** are selected as the **testing data**.

---

##  3. Converting Text to Numbers (TF-IDF)
- The text is converted into numerical form using **TF-IDF (Term Frequency - Inverse Document Frequency)**.
- TF-IDF gives a score to each word based on how often it appears in a thought, and how unique it is across all thoughts.
- This helps the model understand the importance of each word.

---

##  4. Finding Similarities (Cosine Similarity)
- Cosine similarity measures how similar two thoughts are by comparing their TF-IDF vectors.
- A score is given between **0** (no similarity) and **1** (exact match).

---

##  5. Checking Each Test Thought
- For each test thought:
  - It finds the most similar thought from the training data.
  - If the similarity score is **above 0.7**, it is marked as a **known or similar thought**.
  - If the score is **0.7 or below**, it is marked as a **new or different thought**.

---

##  6. Final Result Summary
- The program counts how many test thoughts are **known** (similar to training thoughts).
- It also shows how many are **new or different**.

---
## requirements

install required python libraries:

```bash
pip install pandas scikit-learn
```

##  Final Goal
This method helps you:
- Detect repeated or similar ideas from a large list of thoughts.
- Identify fresh, unique, or different thoughts that don’t match with known ones.

