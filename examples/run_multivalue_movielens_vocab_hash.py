from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names
import functools
import os
import numpy as np
import pandas as pd
import shutil
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
try:
    import tensorflow.compat.v1 as tf
except:
    import tensorflow as tf


def init_vocab(df, tmpdir):
    """initialize the vacabulary file of the sparse features
    """
    vocab_size = {}

    df_user_id = df.user_id.drop_duplicates().dropna().sort_values().reset_index().drop(columns='index')
    df_user_id.index += 1
    df_user_id.to_csv(f'{tmpdir}/user_id.csv', sep=',', index=True, header=False)
    # must set to vocabulary size pluse 1, because 0 is used for miss of has and mask, same below
    vocab_size['user_id'] = len(df_user_id) + 1

    df_movie_id = df.movie_id.drop_duplicates().dropna().sort_values().reset_index().drop(
        columns='index')
    df_movie_id.index += 1
    df_movie_id.to_csv(f'{tmpdir}/movie_id.csv', sep=',', index=True, header=False)
    vocab_size['movie_id'] = len(df_movie_id) + 1

    df_genre = pd.DataFrame({
        'genre': list(set(functools.reduce(lambda x, y: x + y, df.genres.str.split('|'))))
    }).genre.sort_values()
    df_genre.index += 1
    df_genre.to_csv(f'{tmpdir}/genre.csv', sep=',', index=True, header=False)
    vocab_size['genre'] = len(df_genre) + 1

    df_gender = df.gender.drop_duplicates().replace(
        r'^\s*$', np.nan,
        regex=True).dropna().sort_values().reset_index().drop(
            columns='index')
    df_gender.index += 1
    df_gender.to_csv(f'{tmpdir}/gender.csv', sep=',', index=True, header=False)
    vocab_size['gender'] = len(df_gender) + 1

    df_age = df.age.drop_duplicates().dropna().sort_values().reset_index().drop(columns='index')
    df_age.index += 1
    df_age.to_csv(f'{tmpdir}/age.csv', sep=',', index=True, header=False)
    vocab_size['age'] = len(df_age) + 1

    df_occupation = df.occupation.drop_duplicates().replace(
        r'^\s*$', np.nan,
        regex=True).dropna().sort_values().reset_index().drop(
            columns='index')
    df_occupation.index += 1
    df_occupation.to_csv(f'{tmpdir}/occupation.csv', sep=',', index=True, header=False)
    vocab_size['occupation'] = len(df_occupation) + 1

    df_zip = df.zip.drop_duplicates().replace(
        r'^\s*$', np.nan,
        regex=True).dropna().sort_values().reset_index().drop(columns='index')
    df_zip.index += 1
    df_zip.to_csv(f'{tmpdir}/zip.csv', sep=',', index=True, header=False)
    vocab_size['zip'] = len(df_zip) + 1
    return vocab_size


if __name__ == "__main__":
    # change this to where the movielens dataset and work directory is
    workdir = os.path.dirname(__file__)
    data = pd.read_csv(f"{workdir}/movielens_sample.txt")

    metadir = f'{workdir}/meta'
    if not os.path.exists(metadir):
        os.mkdir(metadir)
    vocab_size = init_vocab(data, metadir)

    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip", ]

    data[sparse_features] = data[sparse_features].astype(str)
    target = ['rating']

    # 1.Use hashing encoding on the fly for sparse features,and process sequence features

    genres_list = list(map(lambda x: x.split('|'), data['genres'].values))
    genres_length = np.array(list(map(len, genres_list)))
    max_len = max(genres_length)

    # Notice : padding=`post`
    genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', dtype=str, value=0)

    # 2.set hashing space for each sparse field and generate feature config for sequence feature

    fixlen_feature_columns = [SparseFeat(feat, vocab_size[feat], embedding_dim=4, use_hash=True, vocabulary_path=f'{metadir}/{feat}.csv', dtype='string')
                              for feat in sparse_features]
    varlen_feature_columns = [
        VarLenSparseFeat(SparseFeat('genres', vocabulary_size=vocab_size['genre'], embedding_dim=4, use_hash=True, vocabulary_path=f'{metadir}/genre.csv', dtype="string"),
                         maxlen=max_len, combiner='mean',
                         )]  # Notice : value 0 is for padding for sequence input feature
    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    model_input = {name: data[name] for name in feature_names}
    model_input['genres'] = genres_list

    # 4.Define Model,compile and train
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
    model.compile("adam", "mse", metrics=['mse'], )
    if not hasattr(tf, 'version') or tf.version.VERSION < '2.0.0':
        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            history = model.fit(model_input, data[target].values,
                                batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    else:
        history = model.fit(model_input, data[target].values,
                            batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    if os.path.exists(metadir):
        shutil.rmtree(metadir)

# %%
