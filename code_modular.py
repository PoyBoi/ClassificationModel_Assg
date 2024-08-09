import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load datasets into pandas dataframes
def load_data(file_path):
  """Loads data from a CSV file into a Pandas DataFrame."""
  column_names = ["Abstract", "Domain"]
  df = pd.read_csv(file_path, header=None, names=column_names)
  return df

# Split dataframe into features and labels
def split_data(df):
  """Splits the DataFrame into features (X) and labels (y)."""
  X = df['Abstract']
  y = df['Domain']
  return X, y

# Evaluate the given model using multiple metrics
def evaluate_model(model_name, y_true, y_pred):
    """Evaluates the model and prints various metrics."""
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

# Train and evaluate a single model
def train_and_evaluate_model(model, model_name, X_train, y_train, X_val, y_val):
  """Trains the given model and evaluates it on the validation set."""
  model.fit(X_train, y_train)
  y_pred = model.predict(X_val)
  evaluate_model(model_name, y_val, y_pred)

# Load the datasets
df_train = load_data(r"C:\Users\parvs\Downloads\train.csv")
df_val = load_data(r"C:\Users\parvs\Downloads\validation.csv")

# Prepare the data
X_train, y_train = split_data(df_train)
X_val, y_val = split_data(df_val)

print("Files Read")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print("Starting Vectorizing Data")

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test) 
X_val_vec = vectorizer.transform(X_val)

print("Vectorizing Data Finished\nStarting Model Training")

# 1. Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
train_and_evaluate_model(model_lr, "Logistic Regression (Validation)", X_train_vec, y_train, X_val_vec, y_val)

# 2. Linear Support Vector Machine
model_svm = LinearSVC()
train_and_evaluate_model(model_svm, "Linear SVM (Validation)", X_train_vec, y_train, X_val_vec, y_val)

# 3. Multinomial Naive Bayes
model_nb = MultinomialNB()
train_and_evaluate_model(model_nb, "Multinomial Naive Bayes (Validation)", X_train_vec, y_train, X_val_vec, y_val)

print("Model Training Finished")

# --- Hyperparameter Tuning (Example with GridSearchCV) ---

# Define the parameter grid for Logistic Regression within the pipeline context
param_grid = {
    'lr__C': [0.1, 1, 10],  # Notice 'lr__C' instead of just 'C'
    'lr__penalty': ['l1', 'l2'],
    'lr__solver': ['liblinear'] # Removed Saga because it was more bigger datasets, and lbfgs because it onlt supports L2
}

# Create a pipeline for Logistic Regression
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('lr', LogisticRegression(max_iter=1000))
])

# Create GridSearchCV object
grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_weighted')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Evaluate the best model on the validation set
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_val)
evaluate_model("Best Model (Validation)", y_val, y_pred_best)

print("Finished...")