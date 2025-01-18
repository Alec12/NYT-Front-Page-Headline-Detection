import re
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset with low_memory=False to avoid DtypeWarning
df = pd.read_csv('project/nyt-metadata.csv', low_memory=False)

# Convert print_page to numeric, forcing non-numeric values to NaN
df['print_page'] = pd.to_numeric(df['print_page'], errors='coerce')

# Filter out rows where print_page is NaN
df = df.dropna(subset=['print_page'])

# Create the 'front_page' column
df['front_page'] = df.apply(lambda row: 1 if row['print_section'] == 'A' and row['print_page'] == 1 else 0, axis=1)

features_to_balance = ['section_name', 'type_of_material', 'news_desk']

value_counts = {}
for feature in features_to_balance:
    value_counts[feature] = df[feature].value_counts()

min_target_counts = {feature: value_counts[feature].min() for feature in features_to_balance}

desired_sample_size = 200000

balanced_dataset = pd.DataFrame()

# Ensure balanced sampling for the 'front_page' feature
front_page_counts = df['front_page'].value_counts()
min_count = min(front_page_counts)

balanced_front_page = pd.DataFrame()

for label in front_page_counts.index:
    selected_indices = df[df['front_page'] == label].sample(
        min(min_count, desired_sample_size // len(front_page_counts)), random_state=42
    ).index
    balanced_front_page = pd.concat([balanced_front_page, df.loc[selected_indices]])

balanced_dataset = balanced_front_page.copy()

# Balance the other features as well
for feature in tqdm(features_to_balance):
    unique_values = value_counts[feature].index
    for unique_value in unique_values:
        if len(balanced_dataset) >= desired_sample_size:
            break  # Stop when the desired sample size is reached
        samples_to_select = min_target_counts[feature]
        selected_indices = df[df[feature] == unique_value].sample(
            min(samples_to_select, desired_sample_size - len(balanced_dataset)), random_state=42
        ).index
        balanced_dataset = pd.concat([balanced_dataset, df.loc[selected_indices]])

remaining_samples = desired_sample_size - len(balanced_dataset)
if remaining_samples > 0:
    oversample_indices = df.sample(remaining_samples, random_state=42).index
    balanced_dataset = pd.concat([balanced_dataset, df.loc[oversample_indices]])

print(f"Balanced dataset size: {len(balanced_dataset)}")

balanced_dataset.reset_index(drop=True, inplace=True)

def str_to_dict(row):
    if isinstance(row, str):
        try:
            return ast.literal_eval(row)
        except (ValueError, SyntaxError):
            pass
    return {}

balanced_dataset['headline_dict'] = balanced_dataset['headline'].apply(str_to_dict)
balanced_dataset['keywords_dict'] = balanced_dataset['keywords'].apply(str_to_dict)
balanced_dataset['byline_dict'] = balanced_dataset['byline'].apply(str_to_dict)

balanced_dataset["headline_dict"][0]

def extract_value_from_ldict(d, key):
    if isinstance(d, list):
        items = []
        for dict_ in d:
            if dict_.get('name', None) == key:
                items.append(dict_.get('value', None))
        if not items:
            return np.nan
        return items
    else:
        return np.nan
    
balanced_dataset['organizations'] = balanced_dataset['keywords_dict'].apply(lambda x: extract_value_from_ldict(x, 'organizations'))
balanced_dataset['people'] = balanced_dataset['keywords_dict'].apply(lambda x: extract_value_from_ldict(x, 'persons'))
balanced_dataset['subjects'] = balanced_dataset['keywords_dict'].apply(lambda x: extract_value_from_ldict(x, 'subject'))
balanced_dataset['glocations'] = balanced_dataset['keywords_dict'].apply(lambda x: extract_value_from_ldict(x, 'glocations'))

def extract_value_from_dict(d, key):
    if isinstance(d, dict):
        return d.get(key, np.nan)
    else:
        return np.nan

balanced_dataset['title'] = balanced_dataset['headline_dict'].apply(lambda x: extract_value_from_dict(x, 'main'))
balanced_dataset['kicker'] = balanced_dataset['headline_dict'].apply(lambda x: extract_value_from_dict(x, 'kicker'))
balanced_dataset['author'] = balanced_dataset['byline_dict'].apply(lambda x: extract_value_from_dict(x, 'original')).str.replace('^By ', '', regex=True)

balanced_dataset.drop(['headline', 'keywords', 'byline', 'headline_dict', 'keywords_dict', 'byline_dict'], inplace=True, axis=1)

def extract_numbers(text):
    if isinstance(text, str): 
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])
        else:
            return np.nan
    return text

balanced_dataset['print_page'] = balanced_dataset['print_page'].apply(extract_numbers).astype('Int64')
balanced_dataset['print_page'].unique()

balanced_dataset['pub_date'] = pd.to_datetime(balanced_dataset['pub_date'], errors='coerce')

print('Count before:', len(balanced_dataset['news_desk'].dropna(inplace=False).unique()))

balanced_dataset.loc[balanced_dataset['news_desk'].isin(['', ' ', '0', 'nodesk', 'none']), 'news_desk'] = np.nan
balanced_dataset['news_desk'] = balanced_dataset['news_desk'].str.lower()
news_desk_values = df['news_desk'].dropna(inplace=False).unique()

threshold = 0.8
correct_val = ''

examples = []
replacement_mapping = {}

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(news_desk_values)
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

for i in range(len(similarity_matrix)):
    similar_indices = (similarity_matrix[i] > threshold).nonzero()[0]
    for j in similar_indices:
        if i != j:
            if len(news_desk_values[j]) <= len(news_desk_values[i]):
                replacement_mapping[news_desk_values[i]] = news_desk_values[j]
                if np.random.rand() < 0.1:
                    examples.append(f'{news_desk_values[i]} -> {news_desk_values[j]}')

balanced_dataset['news_desk'].replace(replacement_mapping, inplace=True)

print('Count after:', len(balanced_dataset['news_desk'].dropna(inplace=False).unique()))

balanced_dataset.to_csv('project/nyt-balanced.csv', index=False)
