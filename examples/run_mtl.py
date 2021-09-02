import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import MMOE

if __name__ == "__main__":
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']
    data = pd.read_csv('./census-income.sample', header=None, names=column_names)

    data['label_income'] = data['income_50k'].map({' - 50000.': 0, ' 50000+.': 1})
    data['label_marital'] = data['marital_stat'].apply(lambda x: 1 if x == ' Never married' else 0)
    data.drop(labels=['income_50k', 'marital_stat'], axis=1, inplace=True)

    columns = data.columns.values.tolist()
    sparse_features = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                       'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                       'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                       'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                       'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                       'vet_question']
    dense_features = [col for col in columns if
                      col not in sparse_features and col not in ['label_income', 'label_marital']]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=4) for feat in sparse_features] \
                             + [DenseFeat(feat, 1, ) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    model = MMOE(dnn_feature_columns, tower_dnn_hidden_units=[], task_types=['binary', 'binary'],
                 task_names=['label_income', 'label_marital'])
    model.compile("adam", loss=["binary_crossentropy", "binary_crossentropy"],
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, [train['label_income'].values, train['label_marital'].values],
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=256)

    print("test income AUC", round(roc_auc_score(test['label_income'], pred_ans[0]), 4))
    print("test marital AUC", round(roc_auc_score(test['label_marital'], pred_ans[1]), 4))
