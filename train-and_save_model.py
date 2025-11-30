import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Define feature names
# -----------------------------
FEATURE_NAMES = [
    "age",
    "monthly_charges",
    "tenure_months",
    "num_logins_30_days",
    "num_support_tickets_90_days",
    "is_premium_plan"
]

# -----------------------------
# 2. Generate synthetic numeric data
# -----------------------------
np.random.seed(42)
n_samples = 2000

age = np.random.randint(18, 75, size=n_samples)
monthly_charges = np.random.uniform(20, 200, size=n_samples)
tenure_months = np.random.randint(1, 72, size=n_samples)
num_logins_30_days = np.random.poisson(lam=20, size=n_samples)
num_support_tickets_90_days = np.random.poisson(lam=1.5, size=n_samples)
is_premium_plan = np.random.binomial(n=1, p=0.3, size=n_samples)

X = np.column_stack([
    age,
    monthly_charges,
    tenure_months,
    num_logins_30_days,
    num_support_tickets_90_days,
    is_premium_plan
])

logit = (
    -3.0
    + 0.02 * age
    + 0.03 * monthly_charges
    - 0.04 * tenure_months
    + 0.15 * num_support_tickets_90_days
    - 0.01 * num_logins_30_days
    - 0.3 * is_premium_plan
)

prob = 1 / (1 + np.exp(-logit))
y = (np.random.rand(n_samples) < prob).astype(int)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Training accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

# save model + feature names
joblib.dump(model, "model.pkl")
joblib.dump(FEATURE_NAMES, "feature_names.pkl")

print("Saved model.pkl and feature_names.pkl successfully!")
