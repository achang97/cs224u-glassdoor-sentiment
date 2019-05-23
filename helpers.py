from collections import Counter, namedtuple
from nltk.tree import Tree
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import scipy.stats
import utils
import os

def train_dev_test_split(glassdoor_home, output_var='rating_overall', train_split=0.7, dev_split=0.1, test_split=0.2, \
                         random_state=None, verbose=True):
    src_filename = os.path.join(glassdoor_home, 'glassdoor-data.csv')
    
    cols_to_use = ['date', 'company', 'employee_title', 'review_title', 'pros', 'cons', 'advice_to_mgmt']
    cols_to_use.append(output_var)
    
    # Read and drop NA rows
    data = pd.read_csv(src_filename, usecols=cols_to_use)[cols_to_use]
    data = data.dropna(subset=[output_var])
    data[['employee_title', 'review_title', 'pros', 'cons', 'advice_to_mgmt']] = \
        data[['employee_title', 'review_title', 'pros', 'cons', 'advice_to_mgmt']].fillna('')
    
    data[[output_var]] = data[[output_var]].astype('int')
    
    X = data[cols_to_use[:-1]]
    y = data[[output_var]]
    
    # Split test dataset (20%)
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=test_split, random_state=random_state, stratify=y)
    
    # Then get train and dev (70%, 10%)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=dev_split / (train_split + dev_split), \
                                                      random_state=random_state, stratify=y_train_dev)
    
    if verbose:
        print ("Created splits for {}".format(output_var))
    
    return (X_train, X_dev, X_test, y_train, y_dev, y_test)


def binary_class_func(y):
    """Define a binary SST task.

    Parameters
    ----------
    y : str
        Assumed to be one of the SST labels.

    Returns
    -------
    str or None
        None values are ignored by `build_dataset` and thus left out of
        the experiments.

    """
    if y in (1.0, 2.0):
        return "negative"
    elif y in (4.0, 5.0):
        return "positive"
    else:
        return None


def ternary_class_func(y):
    """Define a binary SST task. Just like `binary_class_func` except
    input '2' returns 'neutral'."""
    if y in (1.0, 2.0):
        return "negative"
    elif y in (4.0, 5.0):
        return "positive"
    else:
        return "neutral"

def build_dataset(X, y, phi, class_func, vectorizer=None, vectorize=True):
    """Core general function for building experimental datasets.
    Returns
    -------
    dict
        A dict with keys 'X' (the feature matrix), 'y' (the list of
        labels), 'vectorizer' (the `DictVectorizer`), and
        'raw_examples' (the `pd.DataFrame` objects, for error analysis).

    """
    labels = [class_func(value) for value in y.values.flatten()]
    raw_examples = [ex for i, ex in X.iterrows()]
    feat_dicts = [phi(ex) for ex in raw_examples]
        
    feat_matrix = None
    if vectorize:
        # In training, we want a new vectorizer:
        if vectorizer == None:
            vectorizer = DictVectorizer(sparse=False)
            feat_matrix = vectorizer.fit_transform(feat_dicts)
        # In assessment, we featurize using the existing vectorizer:
        else:
            feat_matrix = vectorizer.transform(feat_dicts)
    else:
        feat_matrix = feat_dicts
    return {'X': feat_matrix,
            'y': labels,
            'vectorizer': vectorizer,
            'raw_examples': raw_examples}

def experiment(
        phi,
        X_train,
        y_train,
        X_assess,
        y_assess,
        train_func,
        score_func=utils.safe_macro_f1,
        vectorize=True,
        class_func=lambda x: x,
        verbose=True,
        random_state=None):
    """Generic experimental framework. Either assesses with a
    random train/test split of `train_reader` or with `assess_reader` if
    it is given.

    Returns
    -------
    dict with keys
        'model': trained model
        'phi': the function used for featurization
        'train_dataset': a dataset as returned by `build_dataset`
        'assess_dataset': a dataset as returned by `build_dataset`
        'predictions': predictions on the assessment data
        'metric': `score_func.__name__`
        'score': the `score_func` score on the assessment data

    """
    # Train dataset:
    train = build_dataset(
        X_train,
        y_train,
        phi,
        class_func,
        vectorizer=None,
        vectorize=vectorize)
    # Manage the assessment set-up:
    X_train = train['X']
    y_train = train['y']
    raw_train = train['raw_examples']

    # Assessment dataset using the training vectorizer:
    assess = build_dataset(
        X_assess,
        y_assess,
        phi,
        class_func,
        vectorizer=train['vectorizer'],
        vectorize=vectorize)
    X_assess, y_assess = assess['X'], assess['y']
    
    # Train:
    mod = train_func(X_train, y_train)
    # Predictions:
    predictions = mod.predict(X_assess)
    # Report:
    if verbose:
        print(classification_report(y_assess, predictions, digits=3))
    # Return the overall score and experimental info:
    return {
        'model': mod,
        'phi': phi,
        'train_dataset': train,
        'assess_dataset': assess,
        'predictions': predictions,
        'metric': score_func.__name__,
        'score': score_func(y_assess, predictions)}

def compare_models(
        X_train,
        y_train,
        X_assess,
        y_assess,
        phi1,
        train_func1,
        phi2=None,
        train_func2=None,
        vectorize1=True,
        vectorize2=True,
        class_func=lambda x: x,
        stats_test=scipy.stats.wilcoxon,
        trials=10,
        score_func=utils.safe_macro_f1):
    """Wrapper for comparing models. The parameters are like those of
    `experiment`, with the same defaults, except

    Returns
    -------
    (np.array, np.array, float)
        The first two are the scores from each model (length `trials`),
        and the third is the p-value returned by stats_test.

    """
    if phi2 == None:
        phi2 = phi1
    if train_func2 == None:
        train_func2 = train_func1
        
    experiments1 = [experiment(
        phi1,
        X_train,
        y_train,
        X_assess,
        y_assess,
        train_func=train_func1,
        class_func=class_func,
        score_func=score_func,
        vectorize=vectorize1,
        verbose=False) for _ in range(trials)]
    experiments2 = [experiment(
        phi2,
        X_train,
        y_train,
        X_assess,
        y_assess,
        train_func=train_func2,
        class_func=class_func,
        score_func=score_func,
        vectorize=vectorize2,
        verbose=False)  for _ in range(trials)]
    scores1 = np.array([d['score'] for d in experiments1])
    scores2 = np.array([d['score'] for d in experiments2])
    # stats_test returns (test_statistic, p-value). We keep just the p-value:
    pval = stats_test(scores1, scores2)[1]
    # Report:
    print('Model 1 mean: %0.03f' % scores1.mean())
    print('Model 2 mean: %0.03f' % scores2.mean())
    print('p = %0.03f' % pval if pval >= 0.001 else 'p < 0.001')
    # Return the scores for later analysis, and the p value:
    return (scores1, scores2, pval)
