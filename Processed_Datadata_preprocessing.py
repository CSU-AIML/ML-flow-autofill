
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from google.colab import drive
import dvc.api
import os

# Mount Google Drive
drive.mount('/content/drive')

# Load dataset
data_path = '/content/drive/MyDrive/Autofill_App/Raw_data/enhanced_synthetic_data_10k.csv'
df = pd.read_csv(data_path)

# Clean data
df.fillna({'current_maturity': df['current_maturity'].median(), 
           'current_average': df['current_average'].median(),
           'framework': 'Unknown',
           'domain': 'Unknown',
           'gap_description': ''}, inplace=True)

# Encode categorical features
le_framework = LabelEncoder()
le_domain = LabelEncoder()
df['framework_encoded'] = le_framework.fit_transform(df['framework'])
df['domain_encoded'] = le_domain.fit_transform(df['domain'])

# Normalize numerical features
scaler = StandardScaler()
df[['current_average', 'target_average']] = scaler.fit_transform(df[['current_average', 'target_average']])

# Extract features from text
tfidf = TfidfVectorizer(max_features=100)
gap_features = tfidf.fit_transform(df['gap_description']).toarray()
gap_df = pd.DataFrame(gap_features, columns=[f'gap_{i}' for i in range(gap_features.shape[1])])
df = pd.concat([df, gap_df], axis=1)

# Save preprocessed data
preprocessed_path = '/content/drive/MyDrive/Autofill_App/Processed_Data/enhanced_data_preprocessed.csv'
df.to_csv(preprocessed_path, index=False)

# Initialize DVC and version data
os.system('dvc init --no-scm')
os.system(f'dvc add {preprocessed_path}')
os.system('dvc remote add -d myremote gs://your-gcs-bucket/dvc/')
os.system('dvc push')

print("Data preprocessed and versioned with DVC.")
