import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def load_data(file_path):
    """Loads data from CSV."""
    print(f"Loading data from: {file_path}")
    column_names = ["Abstract", "Domain"]
    df = pd.read_csv(file_path, header=None, names=column_names)
    print(f"Data loaded with shape: {df.shape}")
    return df

def split_data(df):
    """Splits DataFrame into features and labels."""
    X = df['Abstract']
    y = df['Domain']
    print(f"Data split into features (X) with shape: {X.shape} and labels (y) with shape: {y.shape}")
    return X, y

def evaluate_model(model_name, y_true, y_pred):
    """Evaluates model and prints metrics."""
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    print(f"--- {model_name} ---")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print("-" * 27)

# Load datasets
df_train = load_data(r"C:\Users\parvs\Downloads\train.csv")
df_val = load_data(r"C:\Users\parvs\Downloads\validation.csv")

# Prepare data
X_train, y_train = split_data(df_train)
X_val, y_val = split_data(df_val)

print("Files Read")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
print(f"Data split into training set with shape: {X_train.shape} and testing set with shape: {X_test.shape}")

# Vectorize text data using TF-IDF
print("Starting Vectorizing Data")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
print("Vectorizing Data Finished")

# Define base models
model_lr = LogisticRegression(max_iter=1000)
model_svm = LinearSVC()
model_nb = MultinomialNB()

# Create Stacking Classifier
print("Creating Stacking Classifier")
estimators = [
    ('lr', model_lr), 
    ('svm', model_svm), 
    ('nb', model_nb)
]
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000)
)

# Train and evaluate Stacking Classifier
print("Training Stacking Classifier")
stacking_clf.fit(X_train_vec, y_train)
print("Evaluating Stacking Classifier")
y_pred_stacking = stacking_clf.predict(X_val_vec)
evaluate_model("Stacking Classifier", y_val, y_pred_stacking)

# ===================================================================
# Output
# ===================================================================

# --- Stacking Classifier ---
# Weighted F1 Score: 0.9117
# Accuracy:          0.9118
# Precision:         0.9118
# Recall:            0.9118
# ---------------------------