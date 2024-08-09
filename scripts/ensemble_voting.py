import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
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
    print("-" * 25)

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
# model_svm = SVC(kernel='linear', probability=True) # Took too long
model_nb = MultinomialNB()

# Create Voting Classifier
print("Creating Voting Classifier")
estimators = [
    ('lr', model_lr), 
    ('svm', model_svm), 
    ('nb', model_nb)
]
voting_clf = VotingClassifier(estimators=estimators, voting='hard') # Using hard voting because soft did not work 
# "'LinearSVC' object has no attribute 'predict_proba'. Did you mean: '_predict_proba_lr'?"

# Train and evaluate Voting Classifier
print("Training Voting Classifier")
voting_clf.fit(X_train_vec, y_train)
print("Evaluating Voting Classifier")
y_pred_voting = voting_clf.predict(X_val_vec)
evaluate_model("Voting Classifier", y_val, y_pred_voting)

# ===================================================================
# Output
# ===================================================================

# --- Voting Classifier ---
# Weighted F1 Score: 0.9099
# Accuracy:          0.9104
# Precision:         0.9098
# Recall:            0.9104
# -------------------------