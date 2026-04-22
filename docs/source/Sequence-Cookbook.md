# Sequence Feature Cookbook

This cookbook collects the input conventions for multi-value features and sequence models such as DIN, BST, DIEN, and DSIN.

## One Row Means One Prediction Target

DeepCTR models expect each training sample to be one row. The row contains:

- the candidate feature to score, such as `item_id` or `cate_id`
- user/context features, such as `user`, `gender`, or `pay_score`
- optional history or session features that describe the user's past behavior
- one label for this candidate row

For example, if the same user has two candidate interactions, build two rows. The history fields can be repeated or updated per row.

```text
user  item_id  cate_id  hist_item_id      hist_cate_id      label
0     1        1        [1, 2, 3, 0]      [1, 2, 2, 0]      1
0     2        2        [1, 2, 3, 1]      [1, 2, 2, 1]      0
```

History fields are context for the row. The label belongs to the candidate row, not to every element inside the history.

## Feature Column Shapes

Use `SparseFeat` for one categorical id per row.

```python
SparseFeat("item_id", vocabulary_size=item_count + 1, embedding_dim=8)
```

Its input shape is usually `(batch_size,)` or `(batch_size, 1)`.

Use `DenseFeat` for numerical values or dense vectors.

```python
DenseFeat("pay_score", 1)
DenseFeat("article_vector", 128)
```

Use `VarLenSparseFeat` for a list of categorical ids per row.

```python
VarLenSparseFeat(
    SparseFeat("genres", vocabulary_size=genre_count + 1, embedding_dim=4),
    maxlen=max_genre_len,
    combiner="mean",
)
```

Its input shape is `(batch_size, maxlen)`. Values must be padded to the same `maxlen` before being passed to the model.

## Padding and Length

For `VarLenSparseFeat`, value `0` is the default padding value. Do not use `0` as a valid category id when `length_name` is not set.

```python
genres = np.array([
    [1, 3, 0, 0],
    [2, 5, 8, 0],
])
```

If you pass `length_name`, DeepCTR uses that length feature to build the pooling mask.

```python
VarLenSparseFeat(
    SparseFeat("hist_item_id", vocabulary_size=item_count + 1, embedding_dim=8),
    maxlen=4,
    length_name="seq_length",
)
```

Then add `seq_length` to the model input:

```python
model_input = {
    "hist_item_id": hist_item_id,
    "seq_length": np.array([3, 2, 4]),
}
```

Padding is still useful because tensors in one batch need the same shape.

## Multi-Value Feature or Multi-Hot Vector

`VarLenSparseFeat` stores categorical ids and then applies embedding lookup plus pooling. `maxlen` is the maximum number of values in one sample. It is not the vocabulary size.

For a `genres` feature with 18 possible genres and at most 5 genres per movie:

```python
VarLenSparseFeat(
    SparseFeat("genres", vocabulary_size=18 + 1, embedding_dim=4),
    maxlen=5,
    combiner="mean",
)
```

A multi-hot vector is also possible, but then it is a dense vector with length equal to the vocabulary size:

```python
DenseFeat("genres_multihot", 18)
```

The two representations are different. `VarLenSparseFeat` learns an embedding for every genre id and pools only the present genres. A multi-hot vector feeds the raw indicator vector into the dense part of the model. For large vocabularies, `VarLenSparseFeat` is usually more memory efficient and easier to share with other categorical fields.

## Multiple VarLenSparseFeat Fields

Add multiple variable-length fields to the same feature column list. Each field can have its own vocabulary and `maxlen`.

```python
feature_columns = [
    SparseFeat("user", user_count + 1, embedding_dim=8),
    SparseFeat("item_id", item_count + 1, embedding_dim=8),
    VarLenSparseFeat(
        SparseFeat("genres", genre_count + 1, embedding_dim=4),
        maxlen=max_genre_len,
        combiner="mean",
    ),
    VarLenSparseFeat(
        SparseFeat("tags", tag_count + 1, embedding_dim=4),
        maxlen=max_tag_len,
        combiner="mean",
    ),
]
```

The model input must include one padded array for each field:

```python
model_input = {
    "user": user,
    "item_id": item_id,
    "genres": genres,
    "tags": tags,
}
```

## Sharing Embeddings

Use the same `embedding_name` when two features represent ids from the same dictionary and should share one embedding table.

```python
feature_columns = [
    SparseFeat("item_id", item_count + 1, embedding_dim=8),
    VarLenSparseFeat(
        SparseFeat(
            "hist_item_id",
            item_count + 1,
            embedding_dim=8,
            embedding_name="item_id",
        ),
        maxlen=4,
    ),
]
```

Use different `embedding_name` values when the fields are semantically different, even if they have the same value range.

## DIN and BST History Feature Names

DIN and BST use `history_feature_list` to find the history sequence that should be matched with the current candidate feature.

```python
behavior_feature_list = ["item_id", "cate_id"]
```

For every name in `behavior_feature_list`, the history sequence feature must be named with the `hist_` prefix:

```python
SparseFeat("item_id", item_count + 1, embedding_dim=8)
SparseFeat("cate_id", cate_count + 1, embedding_dim=4)

VarLenSparseFeat(
    SparseFeat("hist_item_id", item_count + 1, embedding_dim=8, embedding_name="item_id"),
    maxlen=4,
    length_name="seq_length",
)
VarLenSparseFeat(
    SparseFeat("hist_cate_id", cate_count + 1, embedding_dim=4, embedding_name="cate_id"),
    maxlen=4,
    length_name="seq_length",
)
```

If `behavior_feature_list = ["item"]`, the expected history feature name is `hist_item`. If `behavior_feature_list = ["item_id"]`, the expected history feature name is `hist_item_id`.

Inside DIN:

- `varlen_sparse_feature_columns` means all `VarLenSparseFeat` fields passed to the model.
- `history_feature_columns` means the `VarLenSparseFeat` fields whose names match `hist_` + a behavior feature name. These fields are used as keys in attention.
- `sparse_varlen_feature_columns` means other `VarLenSparseFeat` fields. They are pooled and appended to the DNN input, but they are not used as the DIN attention history.

For example, `genres` is a non-history multi-value field:

```python
VarLenSparseFeat(
    SparseFeat("genres", genre_count + 1, embedding_dim=4),
    maxlen=max_genre_len,
    combiner="mean",
)
```

It belongs to `sparse_varlen_feature_columns`, not to `history_feature_columns`.

## DIN Does Not Support VarLen of VarLen

DIN expects the candidate behavior feature to be one sparse id per row, and the history behavior feature to be one padded sequence per row.

Supported:

```text
item_id:       3
hist_item_id:  [1, 2, 4, 0]
```

Not supported directly:

```text
item_categories:       [1, 2, 0]
hist_item_categories:  [[1, 2, 0], [3, 4, 0], [5, 0, 0]]
```

The second case is a "sequence of multi-value behaviors", or a 3D tensor. DeepCTR's current `DIN` implementation does not provide a `VarLenSparseFeat` nested inside another `VarLenSparseFeat`.

Common workarounds are:

- choose one representative category for the candidate and each history item
- map a category set to one categorical id before training
- pre-pool the category set outside the model and build a custom DIN variant
- customize the model to handle a 3D behavior tensor

## Dense History Features in DIN

Current DIN attention is built from sparse embedding sequences. Dense features such as `pay_score` are appended to the DNN input, but they are not part of the attention keys by default.

If you need dense history values inside attention, common choices are:

- discretize the dense value into buckets and use it as a sparse sequence feature
- concatenate or combine dense history representations before passing them to a custom attention layer
- build a custom model based on `deepctr.models.sequence.din.DIN`

DeepCTR does not provide a `VarLenDenseFeat` class.

## DSIN Input Format

DeepCTR's DSIN implementation expects sessions to be prepared before training. It does not split a raw event stream into sessions inside the model.

For `behavior_feature_list = ["item", "cate_id"]` and `sess_max_count=2`, prepare fields like:

```python
feature_columns = [
    SparseFeat("item", item_count + 1, embedding_dim=4),
    SparseFeat("cate_id", cate_count + 1, embedding_dim=4),
    VarLenSparseFeat(
        SparseFeat("sess_0_item", item_count + 1, embedding_dim=4, embedding_name="item"),
        maxlen=4,
    ),
    VarLenSparseFeat(
        SparseFeat("sess_0_cate_id", cate_count + 1, embedding_dim=4, embedding_name="cate_id"),
        maxlen=4,
    ),
    VarLenSparseFeat(
        SparseFeat("sess_1_item", item_count + 1, embedding_dim=4, embedding_name="item"),
        maxlen=4,
    ),
    VarLenSparseFeat(
        SparseFeat("sess_1_cate_id", cate_count + 1, embedding_dim=4, embedding_name="cate_id"),
        maxlen=4,
    ),
]
```

The model input should contain one padded sequence per session field and `sess_length`, the number of valid sessions for each row:

```python
model_input = {
    "item": item,
    "cate_id": cate_id,
    "sess_0_item": sess_0_item,
    "sess_0_cate_id": sess_0_cate_id,
    "sess_1_item": sess_1_item,
    "sess_1_cate_id": sess_1_cate_id,
    "sess_length": np.array([2, 1, 0]),
}
```

Usually you keep only the most recent `sess_max_count` sessions and the most recent `maxlen` events in each session, then pad the rest with `0`.

## DSIN Labels and Sessions

Each row still has one label for the candidate item in that row.

```text
user  item  cate_id  sess_0_item     sess_1_item     sess_length  label
0     1     1        [4, 5, 0, 0]    [2, 3, 0, 0]    2            1
0     2     2        [4, 5, 1, 0]    [2, 3, 0, 0]    2            0
```

The same user can appear in multiple rows. Sessions are historical context for the candidate row. They do not have their own labels inside the DSIN input.

If your raw data contains timestamps or time deltas, use them during preprocessing to split events into sessions. If you want time gaps as model features, bucket them into categorical ids or build a custom model. DeepCTR does not provide `VarLenDenseFeat` for raw variable-length dense time values.

## Estimator and TFRecord Inputs

The Keras-style DeepCTR models use `SparseFeat`, `DenseFeat`, and `VarLenSparseFeat`. The Estimator models use TensorFlow `tf.feature_column` objects directly.

For TFRecord vector features, make the dtype and feature column type match:

- categorical id fields should be integer features and use `categorical_column_with_identity`, `categorical_column_with_hash_bucket`, or another categorical column
- dense float vectors should use `numeric_column(..., shape=(dim,))`, not `embedding_column`

Example:

```python
feature_description = {
    "article_id": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
    "article_vector": tf.io.FixedLenFeature(shape=(128,), dtype=tf.float32),
    "clicked": tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32),
}

dnn_feature_columns = [
    tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity("article_id", num_buckets=100000),
        dimension=8,
    ),
    tf.feature_column.numeric_column("article_vector", shape=(128,)),
]
linear_feature_columns = [
    tf.feature_column.categorical_column_with_identity("article_id", num_buckets=100000),
    tf.feature_column.numeric_column("article_vector", shape=(128,)),
]
```

For padded integer sequence fields in TFRecord, store the sequence as a fixed-length integer feature:

```python
feature_description = {
    "hist_item_id": tf.io.FixedLenFeature(shape=(maxlen,), dtype=tf.int64),
}
```

Then use TensorFlow's categorical feature columns according to your Estimator model. The `VarLenSparseFeat(maxlen=...)` argument is part of the Keras-style API and is not used by `tf.feature_column`.

## Quick Checklist

- Use `0` for padding and start valid categorical ids from `1` for sequence fields.
- Set `vocabulary_size` to at least `max_id + 1`.
- Pad every `VarLenSparseFeat` input to `(batch_size, maxlen)`.
- Use `length_name` when you want explicit sequence lengths.
- Use `embedding_name` to share embeddings between candidate and history fields.
- For DIN/BST, history names must be `hist_` + the names in `behavior_feature_list`.
- For DSIN, split sessions offline and pass `sess_0_*`, `sess_1_*`, and `sess_length`.
- Use `DenseFeat(name, dimension)` or `numeric_column(name, shape=(dimension,))` for dense vectors.
