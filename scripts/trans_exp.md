**1. Importing Libraries:**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
import torch
```

* **pandas** is used for data manipulation and analysis.
* **train_test_split** from **sklearn.model_selection** is used to split the data into training and testing sets.
* **f1_score, accuracy_score, precision_score, recall_score** from **sklearn.metrics** are used for evaluating the performance of the model.
* **AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer** from **transformers** are core components for working with Hugging Face's Transformer models.
* **LabelEncoder** from **sklearn.preprocessing** is used to convert categorical labels (text) into numerical form.
* **torch** is the core library for PyTorch, which we'll use indirectly through the `transformers` library.

**2. Defining Helper Functions:**

```python
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
```

* **`load_data(file_path)`:** This function loads data from a CSV file.
   * It takes the `file_path` as input.
   * Reads the CSV file using pandas, assuming no header and sets column names to "Abstract" and "Domain".
   * Prints the shape (rows, columns) of the loaded DataFrame.
   * Returns the DataFrame. 
* **`split_data(df)`:** This function separates features (text data) and labels (domains).
   * Takes a DataFrame `df` as input.
   * Extracts the "Abstract" column as features (X) and the "Domain" column as labels (y).
   * Prints the shapes of the resulting feature and label arrays.
   * Returns the features (X) and labels (y).
* **`evaluate_model(model_name, y_true, y_pred)`:** This function calculates and prints various evaluation metrics.
   * Takes the `model_name`, true labels (`y_true`), and predicted labels (`y_pred`) as input.
   * Calculates weighted F1-score, accuracy, precision, and recall.
   * Prints the model name and the calculated metrics in a formatted way.

**3. Loading and Preparing Data:**

```python
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
```

* Loads the training and validation datasets using the `load_data` function. You'll need to replace the file paths with the correct ones.
* Splits the training and validation data into features (X) and labels (y) using the `split_data` function.
* Further splits the training data (`X_train`, `y_train`) into training and testing sets using `train_test_split`.
   * `test_size=0.2` means 20% of the data will be used for testing.
   * `random_state=42` ensures consistent splitting for reproducibility.

**4. Encoding Labels:**

```python
# Encode labels 
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)
```

* Creates a `LabelEncoder` object to convert text labels into numerical representations.
* Fits the encoder on the training labels (`y_train`) and transforms them into numerical labels (`y_train_encoded`).
* Uses the same encoder (fitted on training data) to transform validation and testing labels, ensuring consistency in label encoding.

**5. Setting up the Transformer Model:**

```python
# --- Transformers ---
model_name = "bert-base-uncased"  # You can experiment with different models
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
```

* **`model_name = "bert-base-uncased"`:** Specifies the pre-trained BERT model to use. You can choose from various BERT variants or other transformer models.
* **`tokenizer = AutoTokenizer.from_pretrained(model_name)`:** Loads the tokenizer associated with the chosen BERT model. The tokenizer is responsible for converting text into numerical input for the model.
* **`model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))`:**
   * Loads the pre-trained BERT model for sequence classification.
   * `num_labels` is set to the number of unique classes in your dataset (determined by the number of unique values in `label_encoder.classes_`).

**6. Tokenizing the Data:**

```python
# Tokenize data
train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
val_encodings = tokenizer(list(X_val), truncation=True, padding=True)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True)
```

* Uses the loaded tokenizer to process the text data:
   * **`truncation=True`:** Truncates text sequences longer than the model's maximum input length.
   * **`padding=True`:** Pads shorter sequences to the maximum input length.

**7. Creating Dataset Objects:**

```python
# Create dataset objects 
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, y_train_encoded)
val_dataset = Dataset(val_encodings, y_val_encoded)
test_dataset = Dataset(test_encodings, y_test_encoded)
```

* Defines a custom `Dataset` class, which is required by the Hugging Face `Trainer` for efficient data loading and batching.
* The `__getitem__` method fetches an item at a given index, converting data to PyTorch tensors.
* The `__len__` method returns the dataset's length.
* Creates `Dataset` objects for training, validation, and testing data.

**8. Defining Training Arguments:**

```python
# Training arguments 
training_args = TrainingArguments(
    output_dir='./results',           # Output directory 
    num_train_epochs=3,               # Number of training epochs 
    per_device_train_batch_size=16,   # Batch size per device during training 
    per_device_eval_batch_size=64,    # Batch size for evaluation 
    warmup_steps=500,                 # Number of warmup steps 
    weight_decay=0.01,                # Weight decay 
    logging_dir='./logs',             # Directory for storing logs 
    logging_steps=10,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
)
```

* Creates a `TrainingArguments` object to specify various hyperparameters for the training process. 
* **`output_dir`:** Directory where training results will be saved.
* **`num_train_epochs`:** Number of times the model will be trained on the entire training dataset.
* **`per_device_train_batch_size`:** Number of samples used in each training iteration.
* **`per_device_eval_batch_size`:** Batch size during evaluation.
* **`warmup_steps`:** Number of initial steps with a gradually increasing learning rate.
* **`weight_decay`:** Regularization technique to prevent overfitting.
* **`logging_dir`:** Directory for saving training logs.
* **`logging_steps`:** Frequency of logging training information.
* **`evaluation_strategy`:** Evaluate model performance at every specified step.
* **`load_best_model_at_end`:**  Save and load the best performing model based on evaluation metrics.

**9. Creating the Trainer:**

```python
# Define Trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)
```

* Creates a `Trainer` object from the `transformers` library, which simplifies the training process.
* The `Trainer` takes the following arguments:
   * **`model`:** The BERT model we defined earlier.
   * **`args`:** The `TrainingArguments` specifying the training hyperparameters.
   * **`train_dataset`:** The training dataset.
   * **`eval_dataset`:** The validation dataset (used for evaluating the model's performance during training).

**10. Training the Model:**

```python
# Train the model 
trainer.train()
```

* Starts the training process. The `Trainer` handles batching, optimization, and evaluation based on the provided arguments and datasets.

**11. Evaluating on the Test Set:**

```python
# Evaluate on the test set
test_predictions = trainer.predict(test_dataset)
test_preds =  test_predictions.predictions.argmax(-1)

# Evaluate the model 
evaluate_model('BERT', y_test_encoded, test_preds)
```

* Uses the trained model to make predictions on the test dataset using `trainer.predict()`.
* Extracts the predicted class labels from the model's output.
* Calls the `evaluate_model` function to calculate and display the performance metrics of the model on the test dataset.

This code provides a comprehensive workflow for fine-tuning a BERT model on a text classification task. Remember to replace the placeholder file paths and experiment with different BERT variants or training hyperparameters to optimize the model's performance for your specific task. 
