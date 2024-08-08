import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Load the datasets
column_names = ["Abstract", "Domain"]
df_train = pd.read_csv(r"C:\Users\parvs\Downloads\train.csv", header=None, names=column_names)
df_val = pd.read_csv(r"C:\Users\parvs\Downloads\validation.csv", header=None, names=column_names)

# Prepare the data
X = df_train['Abstract']
y = df_train['Domain']

print("Files Read")

# Function to evaluate and print metrics
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

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Starting Vectorizing Data")

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test) 
X_val_vec = vectorizer.transform(df_val['Abstract'])

y_val = df_val['Domain'] 

print("Vectorizing Data Finished\nStarting Model 1 Training")

# 1. Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_vec, y_train)

# Evaluate on Validation Set
y_pred_lr_val = model_lr.predict(X_val_vec)  # Predict on validation data
evaluate_model("Logistic Regression (Validation)", y_val, y_pred_lr_val)

# 2. Linear Support Vector Machine
model_svm = LinearSVC()
model_svm.fit(X_train_vec, y_train)

# Evaluate on Validation Set
y_pred_svm_val = model_svm.predict(X_val_vec)  # Predict on validation data
evaluate_model("Linear SVM (Validation)", y_val, y_pred_svm_val)

# 3. Multinomial Naive Bayes
model_nb = MultinomialNB()
model_nb.fit(X_train_vec, y_train)

# Evaluate on Validation Set
y_pred_nb_val = model_nb.predict(X_val_vec)  # Predict on validation data
evaluate_model("Multinomial Naive Bayes (Validation)", y_val, y_pred_nb_val)

print("Model 3 Training Finished\n Finished...")