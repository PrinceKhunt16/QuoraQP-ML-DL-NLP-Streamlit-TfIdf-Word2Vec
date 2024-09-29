import numpy as np
import pandas as pd
import string
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz
from gensim.models import Word2Vec
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('omw-1.4')

def text_preprocessor(text):
    if pd.isna(text):
        return ''
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def jaccard_similarity(q1, q2):
    a = set(q1.split())
    b = set(q2.split())

    if len(a) == 0 and len(b) == 0:
        return pd.DataFrame([[1.0]], columns=['jaccard'])
    elif len(a) == 0 or len(b) == 0:
        return pd.DataFrame([[0.0]], columns=['jaccard'])

    return pd.DataFrame([[len(a.intersection(b)) / len(a.union(b))]], columns=['jaccard'])

def compute_token_features(q1, q2):
    tokens_q1 = set(q1.split())
    tokens_q2 = set(q2.split())
    cwc = len(tokens_q1.intersection(tokens_q2))
    csc = len(tokens_q1.intersection(tokens_q2).intersection(stopwords.words('english')))
    len_q1, len_q2 = len(tokens_q1), len(tokens_q2)
    
    cwc_min = cwc / min(len_q1, len_q2) if min(len_q1, len_q2) > 0 else 0
    cwc_max = cwc / max(len_q1, len_q2) if max(len_q1, len_q2) > 0 else 0
    csc_min = csc / min(len([word for word in tokens_q1 if word in stopwords.words('english')]), len([word for word in tokens_q2 if word in stopwords.words('english')])) if min(len([word for word in tokens_q1 if word in stopwords.words('english')]), len([word for word in tokens_q2 if word in stopwords.words('english')])) > 0 else 0
    csc_max = csc / max(len([word for word in tokens_q1 if word in stopwords.words('english')]), len([word for word in tokens_q2 if word in stopwords.words('english')])) if max(len([word for word in tokens_q1 if word in stopwords.words('english')]), len([word for word in tokens_q2 if word in stopwords.words('english')])) > 0 else 0
    ctc_min = cwc / min(len_q1, len_q2) if min(len_q1, len_q2) > 0 else 0
    ctc_max = cwc / max(len_q1, len_q2) if max(len_q1, len_q2) > 0 else 0
    
    last_word_eq = int(q1.split()[-1] == q2.split()[-1]) if q1.split() and q2.split() else 0
    first_word_eq = int(q1.split()[0] == q2.split()[0]) if q1.split() and q2.split() else 0

    return pd.DataFrame([[cwc_min, cwc_max, csc_min, csc_max, ctc_min, ctc_max, last_word_eq, first_word_eq]], columns=['cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max', 'last_word_eq', 'first_word_eq'])

def compute_len_features(row):
    q1 = row['question1'] if isinstance(row['question1'], str) else ''
    q2 = row['question2'] if isinstance(row['question2'], str) else ''
    
    len_q1 = len(q1.split())
    len_q2 = len(q2.split())

    mean_len = (len_q1 + len_q2) / 2
    abs_len_diff = abs(len_q1 - len_q2)

    def longest_common_substring_ratio(s1, s2):
        max_length = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                length = 0
                while (i + length < len(s1)) and (j + length < len(s2)) and (s1[i + length] == s2[j + length]):
                    length += 1
                max_length = max(max_length, length)
        return max_length / min(len(s1), len(s2)) if min(len(s1), len(s2)) > 0 else 0

    longest_substr_ratio = longest_common_substring_ratio(q1, q2)

    return pd.DataFrame([[mean_len, abs_len_diff, longest_substr_ratio]],  columns=['mean_len', 'abs_len_diff', 'longest_substr_ratio'])

def compute_fuzzy_features(q1, q2):
    fuzz_ratio = fuzz.ratio(q1, q2)
    fuzz_partial_ratio = fuzz.partial_ratio(q1, q2)
    token_sort_ratio = fuzz.token_sort_ratio(q1, q2)
    token_set_ratio = fuzz.token_set_ratio(q1, q2)

    return pd.DataFrame([[fuzz_ratio, fuzz_partial_ratio, token_sort_ratio, token_set_ratio]],  columns=['fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'token_set_ratio'])

def tfidf_ml(question1, question2):
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    tfidf_q1 = tfidf_vectorizer.transform([question1]).toarray().flatten()
    tfidf_q2 = tfidf_vectorizer.transform([question2]).toarray().flatten()
    
    feature_data = np.concatenate((tfidf_q1, tfidf_q2))
    num_features_q1 = len(tfidf_q1)
    num_features_q2 = len(tfidf_q2)

    column_names = [f'tfidf_q1_{i}' for i in range(num_features_q1)] + [f'tfidf_q2_{i}' for i in range(num_features_q2)]
    
    tfidf_df = pd.DataFrame([feature_data], columns=column_names)

    return tfidf_df

def tfidf_dl(question1, question2):
    with open('models/tfidf_vectorizer_dl.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    tfidf_q1 = tfidf_vectorizer.transform([question1]).toarray().flatten()
    tfidf_q2 = tfidf_vectorizer.transform([question2]).toarray().flatten()
    
    feature_data = np.concatenate((tfidf_q1, tfidf_q2))
    num_features_q1 = len(tfidf_q1)
    num_features_q2 = len(tfidf_q2)

    column_names = [f'tfidf_q1_{i}' for i in range(num_features_q1)] + [f'tfidf_q2_{i}' for i in range(num_features_q2)]
    
    tfidf_df = pd.DataFrame([feature_data], columns=column_names)

    return tfidf_df

def word2vec_ml(question1, question2):
    word2vec_model = Word2Vec.load('models/word2vec_model.model')

    def get_average_w2v(tokens, model):
        words = [word for word in tokens.split() if word in model.wv]
        
        if not words:
            return np.zeros(model.vector_size)  
        
        return np.mean(model.wv[words], axis=0)

    w2v_q1 = get_average_w2v(question1, word2vec_model)
    w2v_q2 = get_average_w2v(question2, word2vec_model)

    feature_data = np.concatenate((w2v_q1, w2v_q2))
    num_features_q1 = len(w2v_q1)
    num_features_q2 = len(w2v_q2)
    column_names = [f'w2v_q1_{i}' for i in range(num_features_q1)] + [f'w2v_q2_{i}' for i in range(num_features_q2)]
    
    w2v_df = pd.DataFrame([feature_data], columns=column_names)

    return w2v_df

def word2vec_dl(question1, question2):
    word2vec_model = Word2Vec.load('models/word2vec_model.model')

    def get_average_w2v(tokens, model):
        words = [word for word in tokens.split() if word in model.wv]
        
        if not words:
            return np.zeros(model.vector_size)  
        
        return np.mean(model.wv[words], axis=0)

    w2v_q1 = get_average_w2v(question1, word2vec_model)
    w2v_q2 = get_average_w2v(question2, word2vec_model)

    feature_data = np.concatenate((w2v_q1, w2v_q2))
    num_features_q1 = len(w2v_q1)
    num_features_q2 = len(w2v_q2)
    column_names = [f'w2v_q1_{i}' for i in range(num_features_q1)] + [f'w2v_q2_{i}' for i in range(num_features_q2)]
    
    w2v_df = pd.DataFrame([feature_data], columns=column_names)

    return w2v_df

def create_feature_row_ml_model(question1, question2):
    q1_processed = text_preprocessor(question1)
    q2_processed = text_preprocessor(question2)
    jaccard = jaccard_similarity(q1_processed, q2_processed)
    token_features = compute_token_features(q1_processed, q2_processed)
    len_features = compute_len_features(pd.Series({'question1': q1_processed, 'question2': q2_processed}))
    fuzzy_features = compute_fuzzy_features(q1_processed, q2_processed)
    tfidf_features = tfidf_ml(question1, question2)
    w2v_features = word2vec_ml(question1, question2)
    feature_row = pd.concat([jaccard, token_features, len_features, fuzzy_features, tfidf_features, w2v_features], axis=1)
    
    return feature_row

def create_feature_row_dl_model(question1, question2):
    q1_processed = text_preprocessor(question1)
    q2_processed = text_preprocessor(question2)
    jaccard = jaccard_similarity(q1_processed, q2_processed)
    token_features = compute_token_features(q1_processed, q2_processed)
    len_features = compute_len_features(pd.Series({'question1': q1_processed, 'question2': q2_processed}))
    fuzzy_features = compute_fuzzy_features(q1_processed, q2_processed)
    tfidf_features = tfidf_dl(question1, question2)
    w2v_features = word2vec_dl(question1, question2)
    feature_row = pd.concat([jaccard, token_features, len_features, fuzzy_features, tfidf_features, w2v_features], axis=1)
    
    return feature_row