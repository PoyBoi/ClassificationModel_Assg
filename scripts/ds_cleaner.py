import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    print(f"Loading data from: {file_path}")
    column_names = ["Abstract", "Domain"]
    df = pd.read_csv(file_path, header=None, names=column_names)
    print(f"Data loaded with shape: {df.shape}")

    df.drop_duplicates(inplace=True)

    df.dropna(subset=["Abstract", "Domain"], inplace=True)

    df['Abstract'] = df['Abstract'].str.replace('[^a-zA-Z0-9\s]', '', regex=True).str.lower()

    # Extra whitespace removal
    df['Abstract'] = df['Abstract'].str.strip()

    # --- Additional Robust Cleaning ---

    df['Abstract'] = df['Abstract'].str.replace('<.*?>', '', regex=True)

    df['Abstract'] = df['Abstract'].str.replace(r'http\S+', '', regex=True)

    # Remove stop words
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    df['Abstract'] = df['Abstract'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    df = df[df['Abstract'].str.split().str.len() > 5] # Adjust the threshold (5) as needed

    print(f"Data cleaned. New shape: {df.shape}")
    return df

#__main__

df_train = r"C:\Users\parvs\Downloads\train.csv"
df_val = r"C:\Users\parvs\Downloads\validation.csv"

df_1 = pd.read_csv(df_train)
df_2 = pd.read_csv(df_val)

print("===================================================================\nBefore\n===================================================================\n")
print("\n========== DF_Train ==========\n")
print(f"Head: \n{df_1.head()}\nDescribe: \n{df_1.describe()}")
print("\n========== DF_Validation ==========\n")
print(f"Head: \n{df_2.head()}\nDescribe: \n{df_2.describe()}")

df_1 = load_and_clean_data(df_train)
df_2 = load_and_clean_data(df_val)

print("===================================================================\nAfter\n===================================================================\n")
print("\n========== DF_Train ==========\n")
print(f"Head: \n{df_1.head()}\nDescribe: \n{df_1.describe()}")
print("\n========== DF_Validation ==========\n")
print(f"Head: \n{df_2.head()}\nDescribe: \n{df_2.describe()}")