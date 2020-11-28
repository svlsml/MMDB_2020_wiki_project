import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import metrics
import random
import json
import time
from sseclient import SSEClient as EventSource

from bloom_filter import BloomFilter
pd.set_option('display.max_colwidth', -1)


def save_json(file_name, json_obj):
    with open(file_name, "a", encoding='utf-8') as f:
        f.write(json_obj + "\n")


def read_data(file_name):
    return pd.read_json(file_name, lines=True)


def flatten_features(df, col_to_parse, new_columns):
    new_column_names = [col_to_parse + '.' + col for col in new_columns]
    df[new_column_names] = df[col_to_parse].apply(pd.Series)[new_columns]
    return df


def create_dummies(df, column_name):
    return df.join(pd.get_dummies(df[column_name], prefix=column_name, prefix_sep='.'))


def clean_data(df):
    df = df.fillna(0)
    # df.id = df.id.astype('int')
    df = df.fillna(0)
    return df


def create_new_columns(df):
    df['edit_length'] = df['length.new'] - df['length.old']
    df['comment_length'] = df.comment.str.len()


# def main():
#     file_name = 'experiment_big.json'
#     wiki_df = read_data(file_name)
#     print(wiki_df.columns)
#     wiki_df = flatten_features(wiki_df, 'length', ['old', 'new'])
#     wiki_df = flatten_features(wiki_df, 'revision', ['old', 'new'])
#     wiki_df = flatten_features(wiki_df, 'meta', ['uri', 'request_id', 'id', 'dt',
#                                                  'domain', 'stream', 'topic', 'partition', 'offset'])
#     print(wiki_df.columns)
#     important_columns = ['namespace', 'user', 'bot', 'minor',
#                          'length.old', 'length.new', 'revision.old', 'revision.new']
#
#     # features = wiki_df.loc[:, ~wiki_df.columns.isin(['bot', 'user'])]
#     features = wiki_df[important_columns]
#     target = wiki_df.loc[:, wiki_df.columns.isin(['bot'])]
#     rf = RandomForestClassifier()
#     rf_model = rf.fit(features, target)
#
#     # get bots from our prediction
#     predicted_bool = rf_model.predict(features)
#     predicted_users = np.array(wiki_df.loc[predicted_bool, ['user']])
#     filter_size = 20000
#     hash_size = 8
#     bf = BloomFilter(filter_size, hash_size, debug=False)
#     bf.train_bloom_filter(predicted_users)


# main()
