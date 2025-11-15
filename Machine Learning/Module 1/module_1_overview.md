# Module 1 — Foundations of Machine Learning

_Prepared by your Senior ML & AI Instructor_

---

## Learning Goals

- Understand the formal definition of machine learning and why it matters now.
- Distinguish among supervised, unsupervised, and reinforcement learning settings.
- Recognize the core components of an end-to-end ML workflow: data, features, models, evaluation, and deployment.
- Grasp the mathematical building blocks that underpin predictive models.
- Implement a minimal working example of a supervised learning pipeline in Python.

---

## 1. What Is Machine Learning?

Arthur Samuel (1959) defined ML as the field of study that gives computers the ability to learn without being explicitly programmed. More formally, Tom Mitchell (1997):

> A computer program is said to learn from experience `E` with respect to some task `T` and performance measure `P` if its performance at `T`, as measured by `P`, improves with experience `E`.

**Key ingredients**

- **Task (`T`)**: The problem being solved (e.g., predicting house prices).
- **Experience (`E`)**: Data available for learning (historical sales records).
- **Performance (`P`)**: Quantitative metric (e.g., mean squared error).

---

## 2. Taxonomy of Learning Paradigms

- **Supervised Learning**: Learn mapping `f: X → Y` from labeled data. Examples: regression (continuous `Y`), classification (categorical `Y`).
- **Unsupervised Learning**: Discover structure in unlabeled data. Examples: clustering, dimensionality reduction.
- **Reinforcement Learning**: Learn policies `π(a|s)` through trial and feedback rewards in sequential decision settings.

### Supervised Learning Objective

For a hypothesis `h_θ(x)` parameterized by `θ`, we minimize empirical risk:

$$
J(θ) = \frac{1}{m} \sum_{i=1}^{m} L\big(h_θ(x^{(i)}), y^{(i)}\big)
$$

where `L` is a choice of loss function (e.g., squared loss).

### Gradient Descent Update

Given learning rate `α`, the iterative update is:

$$
θ := θ - α \nabla_θ J(θ)
$$

---

## 3. The Machine Learning Workflow

1. **Problem Framing**: Translate business/process questions into ML tasks.
2. **Data Acquisition**: Collect raw data, audit provenance, and assess bias.
3. **Data Preparation**: Clean, impute, engineer features, split datasets.
4. **Modeling**: Select algorithms, train, and tune hyperparameters.
5. **Evaluation**: Use validation/test sets and robust metrics.
6. **Deployment & Monitoring**: Package models, integrate with systems, watch drift.

### Train/Validation/Test Split

Ensure honest assessment by splitting data (`train`, `validation`, `test`) with no leakage. Stratify for imbalanced classes.

---

## 4. Mathematical Foundations

### 4.1 Linear Algebra Essentials

- **Vectors** and **matrices** represent feature sets and datasets.
- Dot product: `x · w = \sum_{j=1}^{n} x_j w_j`.
- Matrix multiplication enables batched predictions: `\hat{Y} = XW`.

### 4.2 Probability & Statistics

- Random variable expectation: `\mathbb{E}[X] = \sum_i x_i p(x_i)`.
- Variance: `\text{Var}(X) = \mathbb{E}[(X - \mu)^2]`.
- Bayes’ Theorem informs probabilistic classifiers:

$$
P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}
$$

### 4.3 Optimization

- Convexity guarantees global minima.
- Gradient and Hessian provide direction and curvature information.

---

## 5. Data Preparation Deep Dive

- **Cleaning**: Handle missingness (drop vs. impute), correct anomalies.
- **Feature Engineering**: Encoding categorical variables, scaling (`StandardScaler`, `MinMaxScaler`), combining domain knowledge.
- **Feature Selection**: Filter (correlation), wrapper (RFE), embedded (lasso).
- **Splitting Strategies**: `train_test_split`, k-fold cross-validation for small datasets.

### Example: Mean Normalization Formula

$$
x_j^{(i)} := \frac{x_j^{(i)} - \mu_j}{\sigma_j}
$$

where `μ_j` and `σ_j` are the mean and standard deviation of feature `j`.

---

## 6. Model Families

- **Linear Models**: Linear/Logistic regression, Ridge/Lasso regularization.
- **Non-linear Models**: Decision trees, random forests, gradient boosting.
- **Distance-Based**: k-nearest neighbors.
- **Probabilistic Models**: Naïve Bayes, Gaussian mixtures.

### Regularized Linear Regression Objective

$$
J(θ) = \frac{1}{m} \sum_{i=1}^m \big(h_θ(x^{(i)}) - y^{(i)}\big)^2 + \lambda \lVert θ \rVert_2^2
$$

Regularization parameter `λ` balances fit and generalization.

---

## 7. Evaluation Metrics

- **Regression**: Mean Squared Error (MSE), Root MSE, Mean Absolute Error.
- **Classification**: Accuracy, Precision, Recall, F1-score, ROC AUC.

### Confusion Matrix Terms

- True Positives (`TP`), False Positives (`FP`), True Negatives (`TN`), False Negatives (`FN`).

### Precision & Recall

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}
$$

### F1 Score

$$
F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

---

## 8. Worked Example — Predicting Housing Prices

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Data ingestion
data = pd.read_csv("housing.csv")  # ensure the dataset is available
features = data.drop(columns=["price"])  # supervised label
labels = data["price"]

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# 3. Feature scaling (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 5. Evaluation
predictions = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f"Root Mean Squared Error: {rmse:.2f}")
```

**Instructor Notes**

- Scaling before linear models keeps units comparable.
- Always isolate the test set to get unbiased performance estimates.
- Consider `Pipeline` to avoid leakage when adding further preprocessing.

---

## 9. Interpreting Linear Models

- Coefficients `θ_j` quantify the marginal effect of feature `x_j`.
- Standardize features to compare coefficient magnitudes fairly.
- Use partial dependence plots for non-linear models' interpretability.

### Coefficient Significance (OLS Approximation)

$$
\hat{θ} = (X^T X)^{-1} X^T y
$$

Confidence intervals require assumptions of homoscedastic Gaussian noise.

---

## 10. Practical Tips & Pitfalls

- **Data Leakage**: Guard against using future or target-derived signals during training.
- **Overfitting**: Visualize learning curves; add regularization or collect more data.
- **Bias & Fairness**: Audit demographic parity; consider fairness constraints.
- **Reproducibility**: Fix random seeds, document versions, maintain experiment logs.

---

## 11. Mini Checklist Before Moving Forward

- [ ] Problem statement aligned with measurable metric.
- [ ] Data audit report completed (quality, missingness, bias).
- [ ] Baseline model trained and benchmarked.
- [ ] Next iteration plan documented (feature ideas, model improvements).

---

_End of Module 1 overview. Reach out with questions before starting Module 2 — mastering fundamentals now prevents costly reteaching later._
