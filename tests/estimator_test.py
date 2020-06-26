import pytest
import tensorflow as tf

from deepctr.estimator import WDL
from packaging import version
from .utils import check_model, SAMPLE_SIZE, get_test_data
import numpy as np

# @pytest.mark.parametrize(
#     'sparse_feature_num,dense_feature_num',
#     [(2, 0), (0, 2)#,(2, 2)
#      ]
# )
def test_WDL():
    # if version.parse(tf.__version__) >= version.parse('2.0.0'):
    #     return
    model_name = "WDL"
    sample_size = SAMPLE_SIZE
    # x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
    #                                       dense_feature_num=dense_feature_num,sequence_feature=[])
    a = np.array([1,2,3,4])
    b = np.array([4,3,2,1])
    c = np.array([0.1,0.2,0.3,0.4])
    y = np.array([0,1,0,1])
    dnn_feature_columns= [tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_identity('a',5),4),tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_identity('b',5),4),tf.feature_column.numeric_column('c')]
    linear_feature_columns =[tf.feature_column.categorical_column_with_identity('a',5),tf.feature_column.categorical_column_with_identity('b',5),tf.feature_column.numeric_column('c')]

    model = WDL(linear_feature_columns, dnn_feature_columns,
                dnn_hidden_units=[4, 4], dnn_dropout=0.5)
    input_fn = tf.estimator.inputs.numpy_input_fn({'a':a,'b':b,'c':c},y,shuffle=False)
    model.train(input_fn)
    model.evaluate(input_fn)


if __name__ == "__main__":
    pass
