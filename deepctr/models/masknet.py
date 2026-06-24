# -*- coding:utf-8 -*-
"""
MaskNet

Reference:
    [1] Wang Z, She Q, Zhang J. MaskNet: Introducing Feature-Wise Multiplication to
    CTR Ranking Models by Instance-Guided Mask. DLP-KDD 2021. (https://arxiv.org/abs/2102.07619)
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Multiply, Activation

from ..feature_column import build_input_features, get_linear_logit, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.normalization import LayerNormalization
from ..layers.utils import combined_dnn_input, concat_func, add_func


def _mask_block(emb_in, ln_emb, mask_agg_dim, block_dim):
    """One parallel MaskBlock: an instance-guided multiplicative mask followed by
    a hidden projection + LayerNorm.

    :param emb_in: raw flattened field embeddings (B, D) — drives the mask.
    :param ln_emb: layer-normed embeddings (B, D) — the masked signal.
    """
    feat_dim = int(ln_emb.shape[-1])
    # instance-guided mask: aggregation -> projection back to the feature dim
    mask = Dense(mask_agg_dim, activation="relu")(emb_in)
    mask = Dense(feat_dim)(mask)
    masked = Multiply()([ln_emb, mask])
    hidden = Dense(block_dim, use_bias=False)(masked)
    hidden = LayerNormalization()(hidden)
    return Activation("relu")(hidden)


def MaskNet(linear_feature_columns, dnn_feature_columns, num_blocks=3, block_dim=64,
            mask_agg_dim=None, dnn_hidden_units=(128, 64), l2_reg_linear=0.00001,
            l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
            dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the parallel MaskNet architecture.

    :param linear_feature_columns: features for the linear part.
    :param dnn_feature_columns: features for the deep part.
    :param num_blocks: number of parallel MaskBlocks.
    :param block_dim: output dim of each MaskBlock's hidden projection.
    :param mask_agg_dim: aggregation dim of the instance-guided mask MLP
        (defaults to the embedding feature dim).
    :param dnn_hidden_units: layer sizes of the DNN on top of the concatenated blocks.
    :param task: ``"binary"`` or ``"regression"``.
    :return: A Keras model instance.
    """
    features = build_input_features(linear_feature_columns + dnn_feature_columns)
    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    sparse_embedding_list, dense_value_list = input_from_feature_columns(
        features, dnn_feature_columns, l2_reg_embedding, seed)

    emb_in = combined_dnn_input(sparse_embedding_list, dense_value_list)  # (B, D)
    if mask_agg_dim is None:
        mask_agg_dim = int(emb_in.shape[-1])
    ln_emb = LayerNormalization()(emb_in)

    blocks = [_mask_block(emb_in, ln_emb, mask_agg_dim, block_dim) for _ in range(num_blocks)]
    merged = concat_func(blocks) if len(blocks) > 1 else blocks[0]

    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                  seed=seed)(merged)
    dnn_logit = Dense(1, use_bias=False)(dnn_out)

    final_logit = add_func([linear_logit, dnn_logit])
    output = PredictionLayer(task)(final_logit)

    model = Model(inputs=inputs_list, outputs=output)
    return model
