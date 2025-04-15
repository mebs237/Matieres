import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

seed = 12298374
np.random.seed(seed)

# Simulate data such that Y is independent of X (nothing to learn)
n = 50 # Number of samples
p = 5000 # Number of features
X = np.random.randn(n, p) # Random data
Y = np.random.randint(0, 2, n) # Random binary labels

# Perform feature selection on the entire dataset before cross-validation...
# -> NOT A GOOD PRACTICE; data leakage!!!
selector = SelectKBest(f_classif, k=5) # Select top 5 features
X_selected = selector.fit_transform(X, Y) # Data leakage here

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
cv_errors = []

for train_idx, test_idx in kf.split(X_selected):
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    error = 1 - accuracy_score(Y_test, Y_pred)
    cv_errors.append(error)

print(f"Cross-validation error: {np.mean(cv_errors):.2f}")
