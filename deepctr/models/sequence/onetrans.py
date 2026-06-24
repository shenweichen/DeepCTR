# -*- coding:utf-8 -*-
"""
OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer
in Industrial Recommender Systems.

Architecture
------------
Traditional approaches process feature interaction and sequence modeling in two
separate modules. OneTrans concatenates all feature field tokens and behavior
sequence tokens into a single sequence and feeds them into one Transformer:

    [target_item | feat_1 | ... | feat_F | seq_1 | ... | seq_T]
             ↓  unified Transformer with structured mask  ↓
    [out_0   | out_1   | ... | out_F   | out_s1 | ... | out_sT]
             ↓  flatten feature token outputs  ↓
                          DNN  →  logit

Attention mask:
  - Feature tokens ↔ feature tokens:   fully bidirectional.
  - Feature tokens → sequence tokens:  attend to all valid (non-padding) positions.
  - Sequence token t ← feature tokens: attend to all.
  - Sequence token t ← sequence t':    causal (t' ≤ t) and non-padding.

Key difference vs BST:
  BST runs Transformer only over the sequence, then applies target attention to
  collapse it, and concatenates the result with feature embeddings before DNN.
  OneTrans feeds features AND sequence tokens together so the Transformer can
  model cross-interactions between feature fields and sequence items directly.

Reference:
    OneTrans: Unified Feature Interaction and Sequence Modeling with One
    Transformer in Industrial Recommender Systems.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Concatenate

from ...feature_column import (
    SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features,
)
from ...inputs import (
    create_embedding_matrix, embedding_lookup, get_dense_input,
    varlen_embedding_lookup,
)
from ...layers.core import DNN, PredictionLayer
from ...layers.onetrans import OneTransLayer, SinusoidalPositionEncoding, TokenSlice
from ...layers.utils import concat_func, combined_dnn_input


def OneTrans(
    dnn_feature_columns,
    history_feature_list,
    transformer_num=2,
    att_head_num=4,
    att_embedding_size=None,
    use_ffn=True,
    ffn_expand=4,
    dnn_hidden_units=(256, 128, 64),
    dnn_activation="relu",
    dnn_dropout=0.0,
    dnn_use_bn=False,
    l2_reg_embedding=1e-6,
    l2_reg_dnn=0,
    seed=1024,
    task="binary",
):
    """Instantiates the OneTrans model.

    Parameters
    ----------
    dnn_feature_columns : list
        All feature columns used by the model (SparseFeat, VarLenSparseFeat,
        DenseFeat).  Sequence features must be ``VarLenSparseFeat`` with a
        ``length_name`` pointing to a ``DenseFeat`` that carries the valid
        sequence length for each sample.
    history_feature_list : list of str
        Names of the **target** sparse features whose sequence counterparts
        (prefixed with ``"hist_"``) live in ``VarLenSparseFeat`` columns.
        E.g. ``["item_id"]`` means there is a ``VarLenSparseFeat`` named
        ``"hist_item_id"`` in ``dnn_feature_columns``.
    transformer_num : int
        Number of stacked OneTrans transformer blocks.
    att_head_num : int
        Number of attention heads.
    att_embedding_size : int or None
        Size per attention head.  If None, inferred as
        ``embedding_dim // att_head_num`` from the first feature column.
    use_ffn : bool
        Whether to include the point-wise FFN sub-layer in each block.
    ffn_expand : int
        FFN hidden dim = embedding_dim * ffn_expand.
    dnn_hidden_units : tuple of int
        Layer sizes of the final prediction DNN.
    dnn_activation : str
        Activation for the DNN.
    dnn_dropout : float
        Dropout rate in the DNN and Transformer.
    dnn_use_bn : bool
        Whether to use BatchNorm in the DNN.
    l2_reg_embedding : float
        L2 regularization on embedding tables.
    l2_reg_dnn : float
        L2 regularization on DNN weights.
    seed : int
        Random seed.
    task : str
        ``"binary"`` or ``"regression"``.

    Returns
    -------
    tensorflow.python.keras.Model
    """

    # ── 1. Build raw input tensors ──────────────────────────────────────────
    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    # ── 2. Classify feature columns ─────────────────────────────────────────
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)
    ) if dnn_feature_columns else []

    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)
    ) if dnn_feature_columns else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)
    ) if dnn_feature_columns else []

    # Separate history columns from other varlen columns
    history_fc_names = ["hist_" + name for name in history_feature_list]
    history_feature_columns = []
    other_varlen_columns = []
    for fc in varlen_sparse_feature_columns:
        if fc.name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            other_varlen_columns.append(fc)

    # ── 3. Retrieve sequence length ─────────────────────────────────────────
    # Expect a DenseFeat named "seq_length" carrying the valid history length.
    seq_length = features["seq_length"]          # (B, 1)
    seq_len_max = history_feature_columns[0].maxlen if history_feature_columns else 1

    # ── 4. Create shared embedding tables ──────────────────────────────────
    embedding_dict = create_embedding_matrix(
        dnn_feature_columns, l2_reg_embedding, seed, prefix="", seq_mask_zero=True
    )

    # ── 5. Collect feature field token embeddings ────────────────────────────
    # target-item embeddings: sparse features in history_feature_list
    target_emb_list = embedding_lookup(
        embedding_dict, features, sparse_feature_columns,
        return_feat_list=history_feature_list, to_list=True,
    )
    # other sparse feature embeddings (context, user attributes, etc.)
    other_sparse_emb_list = embedding_lookup(
        embedding_dict, features, sparse_feature_columns,
        mask_feat_list=history_feature_list, to_list=True,
    )
    # pooled other-varlen embeddings (multi-value features, not behavior seq)
    other_varlen_emb_dict = varlen_embedding_lookup(
        embedding_dict, features, other_varlen_columns
    )
    other_varlen_emb_list = [
        v for v in other_varlen_emb_dict.values()
    ]  # each is (B, 1, D) after pooling — we keep the token dim

    # All feature tokens (each is (B, 1, D)):
    feat_emb_list = target_emb_list + other_sparse_emb_list
    num_feat_tokens = len(feat_emb_list)

    # ── 6. Collect behavior sequence token embeddings ────────────────────────
    hist_emb_list = embedding_lookup(
        embedding_dict, features, history_feature_columns,
        return_feat_list=history_fc_names, to_list=True,
    )
    # hist_emb_list: list of (B, T, D) tensors (one per history feature field)
    # Sum across fields to keep dimension D consistent with feature tokens.
    # (Concatenating would give D*n_fields, mismatching feature token dim.)
    from ...layers.utils import add_func
    if len(hist_emb_list) == 1:
        seq_emb = hist_emb_list[0]                           # (B, T, D)
    else:
        seq_emb = add_func(hist_emb_list)                    # (B, T, D) element-wise sum

    # ── 7. Add positional encoding to sequence tokens ───────────────────────
    seq_emb = SinusoidalPositionEncoding()(seq_emb)                    # (B, T, D)

    # ── 8. Determine embedding size & validate heads ────────────────────────
    embedding_size = int(seq_emb.shape[-1])
    if att_embedding_size is None:
        if embedding_size % att_head_num != 0:
            raise ValueError(
                "embedding_size (%d) must be divisible by att_head_num (%d). "
                "Either set att_embedding_size explicitly or adjust att_head_num."
                % (embedding_size, att_head_num)
            )
        att_embedding_size = embedding_size // att_head_num

    # Feature tokens must share the same embedding dimension as seq tokens.
    # concat_func stacks them along axis=1 → (B, 1, D) each.
    feat_tokens = concat_func(feat_emb_list, axis=1)         # (B, F, D)

    # ── 9. Build combined token sequence ────────────────────────────────────
    # Shape: (B, F+T, D)
    all_tokens = concat_func([feat_tokens, seq_emb], axis=1)

    # ── 10. Stack OneTrans blocks ────────────────────────────────────────────
    transformer_out = all_tokens
    for _ in range(transformer_num):
        transformer_out = OneTransLayer(
            num_feat_tokens=num_feat_tokens,
            seq_len_max=seq_len_max,
            att_embedding_size=att_embedding_size,
            head_num=att_head_num,
            use_ffn=use_ffn,
            ffn_expand=ffn_expand,
            dropout_rate=dnn_dropout,
            use_layer_norm=True,
            seed=seed,
        )([transformer_out, seq_length])
        # transformer_out: (B, F+T, D)

    # ── 11. Extract feature token outputs as DNN input ───────────────────────
    # Take only the first F positions (feature tokens) — they now encode
    # cross-interaction with the behavior sequence via the unified Transformer.
    # TokenSlice is a serializable layer (Lambda(lambda ...) cannot round-trip
    # through save_model/load_model).
    feat_out = TokenSlice(num_feat_tokens)(transformer_out)  # (B, F, D)
    feat_out_flat = Flatten()(feat_out)                       # (B, F*D)

    # Dense features (continuous values)
    dense_value_list = get_dense_input(features, dense_feature_columns)

    # ── 12. DNN + prediction ─────────────────────────────────────────────────
    dnn_input = combined_dnn_input([feat_out_flat], dense_value_list)
    dnn_out = DNN(
        dnn_hidden_units, dnn_activation,
        l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
    )(dnn_input)
    final_logit = Dense(1, use_bias=False)(dnn_out)
    output = PredictionLayer(task)(final_logit)

    model = Model(inputs=inputs_list, outputs=output)
    return model
