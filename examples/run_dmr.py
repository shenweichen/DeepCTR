import numpy as np
import pandas as pd
from deepctr.models import DMR
from deepctr.inputs import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names


def get_xy_fd():

    max_len = 50

    # user feature size
    user_size = 1141730  # 用户 ID
    cms_segid_size = 97  # 微群 ID
    cms_group_id_size = 13  # cms_group_id
    final_gender_code_size = 3  # 性别
    age_level_size = 7  # 年龄层次
    pvalue_level_size = 4  # 消费档次
    shopping_level_size = 4  # 购物深度
    occupation_size = 3  # 是否大学生
    new_user_class_level_size = 5  # 城市层级

    # item feature size
    adgroup_id_size = 846812  # 广告单元ID
    cate_size = 12978  # 商品类目
    campaign_id_size = 423437  # 广告计划ID
    customer_size = 255876  # 广告主ID
    brand_size = 461529  # 品牌ID

    # context feature size
    btag_size = 5  # 行为类型
    pid_size = 2  # 资源位

    # embedding size
    main_embedding_size = 32  # 大尺寸
    other_embedding_size = 8  # 小尺寸

    feature_columns = [
        SparseFeat('uid', user_size, embedding_dim=main_embedding_size),
        SparseFeat('mid', adgroup_id_size, embedding_dim=main_embedding_size),
        SparseFeat('cate', cate_size, embedding_dim=main_embedding_size),
        SparseFeat('brand', brand_size, embedding_dim=main_embedding_size),
        SparseFeat('campaign_id',
                   campaign_id_size,
                   embedding_dim=main_embedding_size),
        SparseFeat('customer',
                   customer_size,
                   embedding_dim=main_embedding_size),
        SparseFeat('cms_segid',
                   cms_segid_size,
                   embedding_dim=other_embedding_size),
        SparseFeat('cms_group_id',
                   cms_group_id_size,
                   embedding_dim=other_embedding_size),
        SparseFeat('final_gender_code',
                   final_gender_code_size,
                   embedding_dim=other_embedding_size),
        SparseFeat('age_level',
                   age_level_size,
                   embedding_dim=other_embedding_size),
        SparseFeat('pvalue_level',
                   pvalue_level_size,
                   embedding_dim=other_embedding_size),
        SparseFeat('shopping_level',
                   shopping_level_size,
                   embedding_dim=other_embedding_size),
        SparseFeat('occupation',
                   occupation_size,
                   embedding_dim=other_embedding_size),
        SparseFeat('new_user_class_level',
                   new_user_class_level_size,
                   embedding_dim=other_embedding_size),
        SparseFeat('pid', pid_size, embedding_dim=other_embedding_size),
        DenseFeat('price', 1)
    ]

    feature_columns += [
        VarLenSparseFeat(SparseFeat('his_cate',
                                    cate_size,
                                    embedding_dim=main_embedding_size,
                                    embedding_name='cate'),
                         maxlen=max_len),
        VarLenSparseFeat(SparseFeat('his_brand',
                                    brand_size,
                                    embedding_dim=main_embedding_size,
                                    embedding_name='brand'),
                         maxlen=max_len),
        VarLenSparseFeat(SparseFeat('cont_btag',
                                    btag_size,
                                    embedding_dim=other_embedding_size),
                         maxlen=max_len),
        VarLenSparseFeat(SparseFeat('cont_btag_dm',
                                    btag_size,
                                    embedding_dim=other_embedding_size),
                         maxlen=max_len),
        VarLenSparseFeat(SparseFeat('cont_position',
                                    max_len,
                                    embedding_dim=other_embedding_size),
                         maxlen=max_len),
        VarLenSparseFeat(SparseFeat('cont_position_dm',
                                    max_len,
                                    embedding_dim=other_embedding_size),
                         maxlen=max_len)
    ]

    data = pd.read_csv('.\\alimama_sample.txt')
    cont_btag_mask = list(
        map(lambda x: x.startswith('cont_btag'), data.columns.values))
    his_cate_mask = list(
        map(lambda x: x.startswith('his_cate'), data.columns.values))
    his_brand_mask = list(
        map(lambda x: x.startswith('his_brand'), data.columns.values))

    behavior_feature_list = ['btag', 'cate', 'brand', 'position']
    cont_position = np.tile(np.arange(max_len), (len(data.index), 1))

    feature_dict = {
        'uid': data['uid'],
        'mid': data['mid'],
        'campaign_id': data['campaign_id'],
        'customer': data['customer'],
        'cms_segid': data['cms_segid'],
        'cms_group_id': data['cms_group_id'],
        'final_gender_code': data['final_gender_code'],
        'age_level': data['age_level'],
        'pvalue_level': data['pvalue_level'],
        'shopping_level': data['shopping_level'],
        'occupation': data['occupation'],
        'new_user_class_level': data['new_user_class_level'],
        'pid': data['pid'],
        'cont_btag': data.iloc[:, cont_btag_mask].values,
        'cont_btag_dm': data.iloc[:, cont_btag_mask].values,
        'cate': data['cate_id'],
        'his_cate': data.iloc[:, his_cate_mask].values,
        'brand': data['brand'],
        'his_brand': data.iloc[:, his_brand_mask].values,
        'price': data['price'],
        'cont_position': cont_position,
        'cont_position_dm': cont_position,
    }
    x = {
        name: feature_dict[name]
        for name in get_feature_names(feature_columns)
    }
    y = data['target'].values
    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    model = DMR(feature_columns,
                behavior_feature_list,
                att_weight_normalization=True)
    model.compile('adam',
                  'binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(x,
                        y,
                        batch_size=256,
                        verbose=1,
                        epochs=10,
                        validation_split=0.5)
