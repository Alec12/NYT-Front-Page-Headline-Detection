import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import hstack, csr_matrix
from lime.lime_tabular import LimeTabularExplainer
from joblib import Parallel, delayed
from tqdm import tqdm

# Load the dataset
data = pd.read_csv('project/nyt-balanced.csv')

# Data preprocessing
data['title'].fillna('', inplace=True)
data['lead_paragraph'].fillna('', inplace=True)
data['print_section'].fillna('Unknown', inplace=True)
data['document_type'].fillna('Unknown', inplace=True)
data['news_desk'].fillna('Unknown', inplace=True)
data['section_name'].fillna('Unknown', inplace=True)
data['type_of_material'].fillna('Unknown', inplace=True)
data['author'].fillna('Unknown', inplace=True)
data['word_count'].fillna(data['word_count'].median(), inplace=True)

data['headline_length'] = data['title'].apply(len)
data['lead_paragraph_length'] = data['lead_paragraph'].apply(len)
data['pub_datetime'] = pd.to_datetime(data['pub_date'])
data['pub_hour'] = data['pub_datetime'].dt.hour
data['pub_day'] = data['pub_datetime'].dt.dayofweek
data['pub_day_of_month'] = data['pub_datetime'].dt.day
data['pub_month'] = data['pub_datetime'].dt.month
data['pub_year'] = data['pub_datetime'].dt.year

data = data.drop(columns=['abstract','news_desk', 'kicker', 'organizations', 'people', 'subjects','section_name', 'glocations', 'pub_date', 'pub_datetime', "print_section"])

features = [
    'headline_length', 'lead_paragraph_length', 'pub_hour', 'pub_day',
    'pub_day_of_month', 'pub_month', 'pub_year', 'word_count', 'title',
    'lead_paragraph', 'type_of_material', 'author'
]
X = data[features]
y = data['front_page']

text_features = ['title', 'lead_paragraph']
categorical_features = ['type_of_material', 'author']
numeric_features = [
    'headline_length', 'lead_paragraph_length', 'pub_hour', 'pub_day',
    'pub_day_of_month', 'pub_month', 'pub_year', 'word_count'
]

# Preprocessing pipelines for numeric, categorical and text features
text_transformer = Pipeline(steps=[
    ('vectorizer', TfidfVectorizer(stop_words='english', max_features=1000))
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('text_title', text_transformer, 'title'),
        ('text_lead_paragraph', text_transformer, 'lead_paragraph')
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
print("Sample training data (first 5 rows):\n", X_train.head())

model.fit(X_train, y_train)

print("Model training completed.")

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Transform only numeric and categorical features for LIME
lime_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_train_lime = lime_preprocessor.fit_transform(X_train)
X_test_lime = lime_preprocessor.transform(X_test)

# Convert sparse matrices to dense format in smaller chunks
def sparse_to_dense_chunk(matrix, chunk_size=1000):
    n_rows = matrix.shape[0]
    dense_matrix = []
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        dense_matrix.append(matrix[start:end].toarray())
    return np.vstack(dense_matrix)

X_train_lime_dense = sparse_to_dense_chunk(X_train_lime)
X_test_lime_dense = sparse_to_dense_chunk(X_test_lime)

# Fit LIME explainer on training data
explainer = LimeTabularExplainer(
    training_data=X_train_lime_dense,
    feature_names=numeric_features + lime_preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features).tolist(),
    class_names=['not front page', 'front page'],
    mode='classification'
)

def explain_instance(i):
    exp = explainer.explain_instance(X_test_lime_dense[i], model.named_steps['classifier'].predict_proba, num_features=10, num_samples=500)
    return exp.as_list()

# Parallelize LIME explanations
n_jobs = -1  # Use all available CPU cores
results = Parallel(n_jobs=n_jobs)(delayed(explain_instance)(i) for i in tqdm(range(X_test_lime_dense.shape[0]), desc="Calculating LIME explanations"))

print("LIME explanations calculated.")
