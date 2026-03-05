"""
Train Spam Detection Models and Save to Pickle File
This script trains three models and saves them with properly fitted TF-IDF vectorizer
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 60)
print("SPAM MAIL DETECTION - MODEL TRAINING")
print("=" * 60)

# ─────────────────────────────────────────────
#  STEP 1: LOAD AND CLEAN DATA
# ─────────────────────────────────────────────
print("\n[1/6] Loading data...")
df = pd.read_csv('mail_data.csv')
print(f"✓ Loaded {len(df)} emails")

print("\n[2/6] Cleaning data...")
initial_count = len(df)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"✓ Removed {initial_count - len(df)} duplicates/nulls")
print(f"✓ Final dataset: {len(df)} emails")

# Encode labels
df['Label'] = df['Category'].map({'spam': 0, 'ham': 1})

# ─────────────────────────────────────────────
#  STEP 2: PREPARE FEATURES (TF-IDF)
# ─────────────────────────────────────────────
print("\n[3/6] Creating TF-IDF features...")
X_text = df['Message']
y = df['Label']

# Initialize and FIT the TF-IDF vectorizer
tfidf = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_tfidf = tfidf.fit_transform(X_text)  # This FITS the vectorizer!

print(f"✓ TF-IDF fitted successfully")
print(f"✓ Vocabulary size: {len(tfidf.vocabulary_)}")
print(f"✓ Feature matrix shape: {X_tfidf.shape}")

# ─────────────────────────────────────────────
#  STEP 3: TRAIN-TEST SPLIT
# ─────────────────────────────────────────────
print("\n[4/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# ─────────────────────────────────────────────
#  STEP 4: TRAIN MODELS
# ─────────────────────────────────────────────
print("\n[5/6] Training models...")

# Model 1: Logistic Regression
print("  → Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train)) * 100
lr_test_acc = accuracy_score(y_test, lr_model.predict(X_test)) * 100
print(f"    ✓ Train: {lr_train_acc:.2f}% | Test: {lr_test_acc:.2f}%")

# Model 2: Naive Bayes
print("  → Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_train_acc = accuracy_score(y_train, nb_model.predict(X_train)) * 100
nb_test_acc = accuracy_score(y_test, nb_model.predict(X_test)) * 100
print(f"    ✓ Train: {nb_train_acc:.2f}% | Test: {nb_test_acc:.2f}%")

# Model 3: Linear Regression
print("  → Training Linear Regression...")
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_train_pred = (lin_model.predict(X_train) >= 0.5).astype(int)
lin_test_pred = (lin_model.predict(X_test) >= 0.5).astype(int)
lin_train_acc = accuracy_score(y_train, lin_train_pred) * 100
lin_test_acc = accuracy_score(y_test, lin_test_pred) * 100
print(f"    ✓ Train: {lin_train_acc:.2f}% | Test: {lin_test_acc:.2f}%")

# ─────────────────────────────────────────────
#  STEP 5: SAVE MODEL BUNDLE
# ─────────────────────────────────────────────
print("\n[6/6] Saving models...")

# Prepare accuracies
accuracies = {
    'Logistic Regression': {'train': round(lr_train_acc, 2), 'test': round(lr_test_acc, 2)},
    'Naive Bayes': {'train': round(nb_train_acc, 2), 'test': round(nb_test_acc, 2)},
    'Linear Regression': {'train': round(lin_train_acc, 2), 'test': round(lin_test_acc, 2)}
}

# Prepare dataset info
dataset_info = {
    'total': len(df),
    'spam': (df['Label'] == 0).sum(),
    'ham': (df['Label'] == 1).sum()
}

# Create bundle with all components
bundle = {
    'tfidf': tfidf,  # FITTED TF-IDF vectorizer
    'logistic_regression': lr_model,
    'naive_bayes': nb_model,
    'linear_regression': lin_model,
    'accuracies': accuracies,
    'dataset_info': dataset_info
}

# Save to pickle file
with open('spam_model_new.pkl', 'wb') as f:
    pickle.dump(bundle, f)

print(f"✓ Model saved as 'spam_model_new.pkl'")
print(f"✓ File size: {round(len(pickle.dumps(bundle)) / 1024, 2)} KB")

# ─────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\n📊 Model Performance Summary:")
print("-" * 60)
for model_name, acc in accuracies.items():
    print(f"{model_name:25s} → Train: {acc['train']:6.2f}% | Test: {acc['test']:6.2f}%")
print("-" * 60)

print("\n📁 Dataset Information:")
print(f"  Total Emails: {dataset_info['total']:,}")
print(f"  Spam Emails:  {dataset_info['spam']:,}")
print(f"  Ham Emails:   {dataset_info['ham']:,}")

print("\n✅ Model file 'spam_model_new.pkl' is ready to use!")
print("   Update app.py to load this file instead of 'spam_model.pkl'")
print("\n" + "=" * 60)
