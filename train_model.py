"""
Misinformation Detection and Fact-Checking Prototype
------------------------------------------------------
Steps: 1. Data Preparation and 2. Model Training
"""

import json
from collections import Counter
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE

# -------------------------------------------------------------------------------
# 1. Data Preparation
# -------------------------------------------------------------------------------

data_file = 'politifact_factcheck_data.json'
with open(data_file, 'r') as f:
    sample_data = json.load(f)

texts = [item['statement'] for item in sample_data]

true_verdicts = ['true', 'mostly-true', 'half-true']
misinformation_verdicts = ['mostly-false', 'false', 'pants-fire']

labels = []
for item in sample_data:
    verdict = item['verdict'].lower()
    if verdict in true_verdicts:
        labels.append(0)
    elif verdict in misinformation_verdicts:
        labels.append(1)

label_counts = Counter(labels)
print("Label distribution:", label_counts)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------------------------------
# 2. Model Training (SMOTE + Ensemble + GridSearchCV)
# -------------------------------------------------------------------------------

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

ensemble_model = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(random_state=42)),
    ],
    voting='soft'
)

pipeline = Pipeline([
    ('clf', ensemble_model)
])

param_grid = {
    'clf__lr__C': [0.1, 1, 10],
    'clf__lr__solver': ['lbfgs', 'liblinear'],
    'clf__rf__n_estimators': [100, 200, 300],
    'clf__rf__max_depth': [None, 10, 20],
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    error_score='raise'
)

grid.fit(X_resampled, y_resampled)
best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)

joblib.dump(best_model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')