import pandas as pd
import numpy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from deepctr.models import DeepFM
from deepctr import SingleFeat
import tensorflow as tf
from keras.callbacks import EarlyStopping

def model_pool(defaultfilename='./input/final_track2_train.txt', defaulttestfile='./input/final_track2_test_no_anwser.txt',
                defaultcolumnname=['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'],
                defaulttarget=['finish', 'like'], defaultmodel="AFM", PERCENT=1):
    data = pd.read_csv(defaultfilename, sep='\t', names=defaultcolumnname, iterator=True)
    #1 train file
    take=[]
    loop = True
    while loop:
        try:
            chunk=data.get_chunk(10000)
            chunk=chunk.take(list(range(PERCENT*100)), axis=0)
            take.append(chunk)
        except StopIteration:
            loop=False
            print('stop iteration')
            
    data = pd.concat(take, ignore_index=True) 
    train_size = data.shape[0]
    print(train_size)
    
    #2 test file       
    test_data = pd.read_csv(defaulttestfile, sep='\t', names=defaultcolumnname, )
    data = data.append(test_data)
    test_size=test_data.shape[0]
    print(test_size)
    
    sparse_features=[]
    dense_features=[]
    target=defaulttarget
    for column in data.columns:
        if column in defaulttarget:
            continue
        if data[column].dtype in  [numpy.float_ , numpy.float64]:
            dense_features.append(column)
        if data[column].dtype in [numpy.int_, numpy.int64]:
            sparse_features.append(column)
    
    #3. Remove na values
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0,)
    #4. Label Encoding for sparse features, and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    #5. Dense normalize
    if dense_features:
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])
    #6. generate input data for model
    sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
                           for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0)
                          for feat in dense_features]
    #7. generate input data for model
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]

    #8.generate data
    print(train.columns)
    train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
        [train[feat.name].values for feat in dense_feature_list]
    test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
        [test[feat.name].values for feat in dense_feature_list]
        
    
    train_labels = [train[target].values for target in defaulttarget]
    test_labels = test[target]

    # 6.choose a model
    import pkgutil
    import mdeepctr.models
#     modelnames = [name for _, name, _ in pkgutil.iter_modules(mdeepctr.__path__)]
#     modelname = input("choose a model: "+",".join(modelnames)+"\n")
#     if not modelname:
    modelname=defaultmodel
    # 7.build a model
    model = getattr(mdeepctr.models, modelname)({"sparse": sparse_feature_list,
                    "dense": dense_feature_list}, final_activation='sigmoid', output_dim=len(defaulttarget))
    # 8. eval predict
    def auc(y_true, y_pred):
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
    
    model.compile("adagrad", loss="binary_crossentropy", metrics=[auc])
    my_callbacks = [EarlyStopping(monitor='loss', min_delta=1e-2, patience=3, verbose=1, mode='min')]
    
    history = model.fit(train_model_input, train_labels,
                        batch_size=4096, epochs=100, verbose=1, callbacks=my_callbacks)
    pred_ans = model.predict(test_model_input, batch_size=2**14)
    
#     nsamples, nx, ny = numpy.asarray(pred_ans).shape
#     pred_ans = numpy.asarray(pred_ans).reshape((nx*ny, nsamples))
#     print(test_labels.shape)
#     print(pred_ans.shape)
#     
#     logloss = round(log_loss(test_labels, pred_ans), 4)
#     try:
#         roc_auc = round(roc_auc_score(test_labels, pred_ans), 4)
#     except:
#         roc_auc=0
        
    result = test_data[['uid', 'item_id', 'finish', 'like']].copy()
    result.rename(columns={'finish': 'finish_probability',
                           'like': 'like_probability'}, inplace=True)
    result['finish_probability'] = pred_ans[0]
    result['like_probability'] = pred_ans[1]
    output = "%s-result.csv" % (modelname)
    result[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv(
        output, index=None, float_format='%.6f')
    
    return history

if __name__ == "__main__":
    import pkgutil
    import mdeepctr.models
    modelnames = [name for _, name, _ in pkgutil.iter_modules(mdeepctr.models.__path__)]
    functions = ["AFM", "DCN", "MLR",  "DeepFM",
           "MLR", "NFM", "DIN", "FNN", "PNN", "WDL", "xDeepFM", "AutoInt", ]
    models_dic = dict((function.lower(),function) for function in functions)
    for modelname in modelnames:
        print(modelname)
        if modelname in ["DIN","WDL"]:
            continue
        history = model_pool(defaultmodel=models_dic[modelname],PERCENT=100)
        print(history.history)
        