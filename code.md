# Spam Classifier Workflow Implementation

This follows the workflow from `spam_classifier_workflow`:

1. Load and inspect dataset
2. EDA: class balance, text length, missing values, frequent words
3. Data preprocessing: lowercasing, punctuation/number cleanup
4. Tokenization with spaCy + stopword removal + lemmatization
5. Train/test split
6. Feature engineering: TF-IDF
7. Model selection: Naive Bayes, Logistic Regression, Random Forest
8. Hyperparameter tuning
9. Final evaluation
10. Prediction on new messages

```python
import re
import string
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
import spacy

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# spaCy tokenizer (tokenization before embedding)
nlp = spacy.blank("en")


# ---------------------------------
# 1. LOAD DATA
# ---------------------------------
df = pd.read_csv("dataset/combined_data.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())


# ---------------------------------
# 2. EDA
# ---------------------------------
print("\nMissing values:\n", df.isnull().sum())
print("\nClass distribution:\n", df["label"].value_counts())

# Text length features
df["text"] = df["text"].astype(str)
df["char_len"] = df["text"].str.len()
df["word_len"] = df["text"].str.split().apply(len)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(x="label", data=df, ax=axes[0])
axes[0].set_title("Class Distribution (0 = Ham, 1 = Spam)")

sns.kdeplot(data=df, x="word_len", hue="label", fill=True, common_norm=False, alpha=0.4, ax=axes[1])
axes[1].set_title("Word Length Distribution by Class")

plt.tight_layout()
plt.show()


# Frequent words per class
def top_words(text_series, n=20):
	joined = " ".join(text_series.tolist())
	tokens = re.findall(r"[a-zA-Z']+", joined.lower())
	tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 2]
	return Counter(tokens).most_common(n)


ham_top = top_words(df.loc[df["label"] == 0, "text"], n=20)
spam_top = top_words(df.loc[df["label"] == 1, "text"], n=20)

ham_df = pd.DataFrame(ham_top, columns=["word", "count"])
spam_df = pd.DataFrame(spam_top, columns=["word", "count"])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(data=ham_df, x="count", y="word", ax=axes[0], color="#4C78A8")
axes[0].set_title("Top Frequent Words in Ham")

sns.barplot(data=spam_df, x="count", y="word", ax=axes[1], color="#F58518")
axes[1].set_title("Top Frequent Words in Spam")

plt.tight_layout()
plt.show()


# ---------------------------------
# 3. PREPROCESSING
# ---------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
	text = text.lower()
	text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove URLs
	text = re.sub(r"\d+", " ", text)  # remove numbers
	text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
	text = re.sub(r"\s+", " ", text).strip()
	return text


# ---------------------------------
# 4. SPACY TOKENIZATION
# ---------------------------------
def tokenize_with_spacy(text: str) -> str:
	tokens = [tok.text for tok in nlp(text) if not tok.is_space]
	tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
	tokens = [lemmatizer.lemmatize(w) for w in tokens]
	return " ".join(tokens)


df["preprocessed_text"] = df["text"].apply(clean_text)
df["tokenized_text"] = df["preprocessed_text"].apply(tokenize_with_spacy)

print("\nSample cleaned text:")
print(df[["text", "preprocessed_text", "tokenized_text"]].head(3))


# ---------------------------------
# 5. TRAIN / TEST SPLIT
# ---------------------------------
X = df["tokenized_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain size:", X_train.shape[0], "Test size:", X_test.shape[0])


# ---------------------------------
# 6. TF-IDF EMBEDDING + MODEL SELECTION
# ---------------------------------
tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

models = {
	"naive_bayes": MultinomialNB(),
	"logistic_regression": LogisticRegression(max_iter=2000, n_jobs=-1),
	"random_forest": RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1),
}

results = []
for model_name, model in models.items():
	model.fit(X_train_vec, y_train)
	preds = model.predict(X_test_vec)

	acc = accuracy_score(y_test, preds)
	f1 = f1_score(y_test, preds)
	prec = precision_score(y_test, preds)
	rec = recall_score(y_test, preds)

	results.append(
		{
			"embedding": "tfidf",
			"model": model_name,
			"accuracy": acc,
			"f1": f1,
			"precision": prec,
			"recall": rec,
		}
	)


results_df = pd.DataFrame(results).sort_values(by=["f1", "accuracy"], ascending=False)
print("\nModel comparison:")
print(results_df)


# ---------------------------------
# 7. HYPERPARAMETER TUNING (BEST BASELINE)
# ---------------------------------
best_row = results_df.iloc[0]
best_model_name = best_row["model"]

print(f"\nBest baseline: {best_model_name} + tfidf")

X_train_best = tfidf.fit_transform(X_train)
X_test_best = tfidf.transform(X_test)

if best_model_name == "naive_bayes":
	param_grid = {"alpha": [0.1, 0.3, 0.5, 1.0]}
	estimator = MultinomialNB()
elif best_model_name == "logistic_regression":
	param_grid = {"C": [0.1, 0.5, 1.0, 2.0], "solver": ["liblinear", "lbfgs"]}
	estimator = LogisticRegression(max_iter=3000, n_jobs=-1)
else:
	param_grid = {
		"n_estimators": [200, 300],
		"max_depth": [None, 20],
		"min_samples_split": [2, 5],
	}
	estimator = RandomForestClassifier(random_state=42, n_jobs=-1)

grid = GridSearchCV(
	estimator=estimator,
	param_grid=param_grid,
	scoring="f1",
	cv=3,
	n_jobs=-1,
	verbose=1,
)

grid.fit(X_train_best, y_train)
best_model = grid.best_estimator_

print("Best params:", grid.best_params_)


# ---------------------------------
# 8. FINAL EVALUATION
# ---------------------------------
final_preds = best_model.predict(X_test_best)

print("\nFinal Metrics")
print("Accuracy:", round(accuracy_score(y_test, final_preds), 4))
print("Precision:", round(precision_score(y_test, final_preds), 4))
print("Recall:", round(recall_score(y_test, final_preds), 4))
print("F1:", round(f1_score(y_test, final_preds), 4))

print("\nClassification Report:\n")
print(classification_report(y_test, final_preds, digits=4))

cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ---------------------------------
# 9. PREDICTION ON NEW MESSAGES
# ---------------------------------
def predict_message(message: str) -> str:
	preprocessed = clean_text(message)
	tokenized = tokenize_with_spacy(preprocessed)
	vectorized = tfidf.transform([tokenized])
	pred = best_model.predict(vectorized)[0]
	return "Spam" if pred == 1 else "Ham"


sample_messages = [
	"Congratulations! You have won a free iPhone. Click here to claim now.",
	"Can we reschedule our meeting to tomorrow afternoon?",
	"Limited offer: Buy medicine online at 70% discount.",
]

print("\nSample predictions:")
for msg in sample_messages:
	label = predict_message(msg)
	print(f"Text: {msg}\nPrediction: {label}\n")
```

## Notes

1. This code uses spaCy tokenization before TF-IDF embedding and compares multiple classical models.
2. If you want, we can add Word2Vec/GloVe or LSTM next as an advanced extension.
