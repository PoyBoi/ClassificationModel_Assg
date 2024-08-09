import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def load_data(file_path):
    print(f"Loading data from: {file_path}")
    column_names = ["Abstract", "Domain"]
    df = pd.read_csv(file_path, header=None, names=column_names)
    print(f"Data loaded with shape: {df.shape}")
    return df

def split_data(df):
    X = df['Abstract']
    y = df['Domain']
    print(f"Data split into features (X) with shape: {X.shape} and labels (y) with shape: {y.shape}")
    return X, y

def evaluate_model(model_name, y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    print(f"--- {model_name} ---")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print("-" * 20)

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

# --- Bagging (Random Forest) ---
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_vec, y_train)
y_pred_rf = rf_clf.predict(X_val_vec)
evaluate_model("Random Forest (Bagging)", y_val, y_pred_rf)

# --- Boosting (Gradient Boosting) ---
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train_vec, y_train)
y_pred_gb = gb_clf.predict(X_val_vec)
evaluate_model("Gradient Boosting", y_val, y_pred_gb)

# ===================================================================
# Output (Bagging AND Boosting)
# ===================================================================

# --- Random Forest (Bagging) ---
# Weighted F1 Score: 0.8514
# Accuracy:          0.8544
# Precision:         0.8537
# Recall:            0.8544
# --------------------
# --- Gradient Boosting ---
# Weighted F1 Score: 0.8648
# Accuracy:          0.8654
# Precision:         0.8646
# Recall:            0.8654
# --------------------