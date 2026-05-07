import pandas as pd
import tensorflow as tf
import json
import requests 
import grpc
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.estimator.inputs import input_fn_pandas
from deepctr.estimator import DeepFMEstimator

from __future__ import print_function
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

def get_integer_mapping(le):
    '''
    Return a dict mapping labels to their integer values from an SKlearn LabelEncoder
    le = a fitted SKlearn LabelEncoder
    '''
    res = {}
    for idx, val in enumerate(le.classes_):
        res.update({val:idx})
    return res


def test_REST_serving():
    '''
    test rest api 
    '''
    fea_dict1 = {'I1':[0.0],'I2':[0.001332],'I3':[0.092362],'I4':[0.0],'I5':[0.034825],'I6':[0.0],'I7':[0.0],'I8':[0.673468],'I9':[0.0],'I10':[0.0],'I11':[0.0],'I12':[0.0],'I13':[0.0],'C1':[0],'C2':[4],'C3':[96],'C4':[146],'C5':[1],'C6':[4],'C7':[163],'C8':[1],'C9':[1],'C10':[72],'C11':[117],'C12':[127],'C13':[157],'C14':[7],'C15':[127],'C16':[126],'C17':[8],'C18':[66],'C19':[0],'C20':[0],'C21':[3],'C22':[0],'C23':[1],'C24':[96],'C25':[0],'C26':[0]}
    fea_dict2 = {'I1':[0.0],'I2':[0.0],'I3':[0.00675],'I4':[0.402298],'I5':[0.059628],'I6':[0.117284],'I7':[0.003322],'I8':[0.714284],'I9':[0.154739],'I10':[0.0],'I11':[0.03125],'I12':[0.0],'I13':[0.343137],'C1':[11],'C2':[1],'C3':[98],'C4':[98],'C5':[1],'C6':[6],'C7':[179],'C8':[0],'C9':[1],'C10':[89],'C11':[58],'C12':[97],'C13':[79],'C14':[7],'C15':[72],'C16':[26],'C17':[7],'C18':[52],'C19':[0],'C20':[0],'C21':[47],'C22':[0],'C23':[7],'C24':[112],'C25':[0],'C26':[0]}
    fea_dict3 = {'I1':[0.0],'I2':[0.000333],'I3':[0.00071],'I4':[0.137931],'I5':[0.003968],'I6':[0.077873],'I7':[0.019934],'I8':[0.714284],'I9':[0.505803],'I10':[0.0],'I11':[0.09375],'I12':[0.0],'I13':[0.17647],'C1':[0],'C2':[18],'C3':[39],'C4':[52],'C5':[3],'C6':[4],'C7':[140],'C8':[2],'C9':[1],'C10':[93],'C11':[31],'C12':[122],'C13':[16],'C14':[7],'C15':[129],'C16':[97],'C17':[8],'C18':[49],'C19':[0],'C20':[0],'C21':[25],'C22':[0],'C23':[6],'C24':[53],'C25':[0],'C26':[0]}
    fea_dict4 = {'I1':[0.0],'I2':[0.004664],'I3':[0.000355],'I4':[0.045977],'I5':[0.033185],'I6':[0.094967],'I7':[0.016611],'I8':[0.081632],'I9':[0.028046],'I10':[0.0],'I11':[0.0625],'I12':[0.0],'I13':[0.039216],'C1':[0],'C2':[45],'C3':[7],'C4':[117],'C5':[1],'C6':[0],'C7':[164],'C8':[1],'C9':[0],'C10':[20],'C11':[61],'C12':[104],'C13':[36],'C14':[1],'C15':[43],'C16':[43],'C17':[8],'C18':[37],'C19':[0],'C20':[0],'C21':[156],'C22':[0],'C23':[0],'C24':[32],'C25':[0],'C26':[0]}
    fea_dict5 = {'I1':[0.0],'I2':[0.000333],'I3':[0.036945],'I4':[0.310344],'I5':[0.003922],'I6':[0.067426],'I7':[0.013289],'I8':[0.65306],'I9':[0.035783],'I10':[0.0],'I11':[0.03125],'I12':[0.0],'I13':[0.264706],'C1':[0],'C2':[11],'C3':[59],'C4':[77],'C5':[1],'C6':[5],'C7':[18],'C8':[1],'C9':[1],'C10':[45],'C11':[171],'C12':[162],'C13':[96],'C14':[4],'C15':[36],'C16':[121],'C17':[8],'C18':[14],'C19':[5],'C20':[3],'C21':[9],'C22':[0],'C23':[0],'C24':[5],'C25':[1],'C26':[47]}

    # json str
    data = json.dumps({"signature_name": "serving_default","instances": [fea_dict1,fea_dict2,fea_dict3,fea_dict4,fea_dict5] })
    # print(data)

    json_response = requests.post('http://localhost:8501/v1/models/raw_export_deepfm_model:predict', data=data)
    predictions = json.loads(json_response.text)
    print(predictions)


def test_RPC_serving():
    '''
    test RPC api 
    '''
    channel = grpc.insecure_channel(target='0.0.0.0:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'raw_export_deepfm_model'
    request.model_spec.signature_name = 'serving_default'

    request.inputs['I1'].CopyFrom(tf.contrib.util.make_tensor_proto([0.0,0.0,0.0,0.0,0.0], shape=[5]))
    request.inputs['I2'].CopyFrom(tf.contrib.util.make_tensor_proto([0.001332,0.0,0.000333,0.004664,0.000333], shape=[5]))
    request.inputs['I3'].CopyFrom(tf.contrib.util.make_tensor_proto([0.092362,0.00675,0.00071,0.000355,0.036945], shape=[5]))
    request.inputs['I4'].CopyFrom(tf.contrib.util.make_tensor_proto([0.0,0.402298,0.137931,0.045977,0.310344], shape=[5]))
    request.inputs['I5'].CopyFrom(tf.contrib.util.make_tensor_proto([0.034825,0.059628,0.003968,0.033185,0.003922], shape=[5]))
    request.inputs['I6'].CopyFrom(tf.contrib.util.make_tensor_proto([0.0,0.117284,0.077873,0.094967,0.067426], shape=[5]))
    request.inputs['I7'].CopyFrom(tf.contrib.util.make_tensor_proto([0.0,0.003322,0.019934,0.016611,0.013289], shape=[5]))
    request.inputs['I8'].CopyFrom(tf.contrib.util.make_tensor_proto([0.673468,0.714284,0.714284,0.081632,0.65306], shape=[5]))
    request.inputs['I9'].CopyFrom(tf.contrib.util.make_tensor_proto([0.0,0.154739,0.505803,0.028046,0.035783], shape=[5]))
    request.inputs['I10'].CopyFrom(tf.contrib.util.make_tensor_proto([0.0,0.0,0.0,0.0,0.0], shape=[5]))
    request.inputs['I11'].CopyFrom(tf.contrib.util.make_tensor_proto([0.0,0.03125,0.09375,0.0625,0.03125], shape=[5]))
    request.inputs['I12'].CopyFrom(tf.contrib.util.make_tensor_proto([0.0,0.0,0.0,0.0,0.0], shape=[5]))
    request.inputs['I13'].CopyFrom(tf.contrib.util.make_tensor_proto([0.0,0.343137,0.17647,0.039216,0.264706], shape=[5]))

    request.inputs['C1'].CopyFrom(tf.contrib.util.make_tensor_proto([0,11,0,0,0], shape=[5]))
    request.inputs['C2'].CopyFrom(tf.contrib.util.make_tensor_proto([4,1,18,45,11], shape=[5]))
    request.inputs['C3'].CopyFrom(tf.contrib.util.make_tensor_proto([96,98,39,7,59], shape=[5]))
    request.inputs['C4'].CopyFrom(tf.contrib.util.make_tensor_proto([146,98,52,117,77], shape=[5]))
    request.inputs['C5'].CopyFrom(tf.contrib.util.make_tensor_proto([1,1,3,1,1], shape=[5]))
    request.inputs['C6'].CopyFrom(tf.contrib.util.make_tensor_proto([4,6,4,0,5], shape=[5]))
    request.inputs['C7'].CopyFrom(tf.contrib.util.make_tensor_proto([163,179,140,164,18], shape=[5]))
    request.inputs['C8'].CopyFrom(tf.contrib.util.make_tensor_proto([1,0,2,1,1], shape=[5]))
    request.inputs['C9'].CopyFrom(tf.contrib.util.make_tensor_proto([1,1,1,0,1], shape=[5]))
    request.inputs['C10'].CopyFrom(tf.contrib.util.make_tensor_proto([72,89,93,20,45], shape=[5]))
    request.inputs['C11'].CopyFrom(tf.contrib.util.make_tensor_proto([117,58,31,61,171], shape=[5]))
    request.inputs['C12'].CopyFrom(tf.contrib.util.make_tensor_proto([127,97,122,104,162], shape=[5]))
    request.inputs['C13'].CopyFrom(tf.contrib.util.make_tensor_proto([157,79,16,36,96], shape=[5]))
    request.inputs['C14'].CopyFrom(tf.contrib.util.make_tensor_proto([7,7,7,1,4], shape=[5]))
    request.inputs['C15'].CopyFrom(tf.contrib.util.make_tensor_proto([127,72,129,43,36], shape=[5]))
    request.inputs['C16'].CopyFrom(tf.contrib.util.make_tensor_proto([126,26,97,43,121], shape=[5]))
    request.inputs['C17'].CopyFrom(tf.contrib.util.make_tensor_proto([8,7,8,8,8], shape=[5]))
    request.inputs['C18'].CopyFrom(tf.contrib.util.make_tensor_proto([66,52,49,37,14], shape=[5]))
    request.inputs['C19'].CopyFrom(tf.contrib.util.make_tensor_proto([0,0,0,0,5], shape=[5]))
    request.inputs['C20'].CopyFrom(tf.contrib.util.make_tensor_proto([0,0,0,0,3], shape=[5]))
    request.inputs['C21'].CopyFrom(tf.contrib.util.make_tensor_proto([3,47,25,156,9], shape=[5]))
    request.inputs['C22'].CopyFrom(tf.contrib.util.make_tensor_proto([0,0,0,0,0], shape=[5]))
    request.inputs['C23'].CopyFrom(tf.contrib.util.make_tensor_proto([1,7,6,0,0], shape=[5]))
    request.inputs['C24'].CopyFrom(tf.contrib.util.make_tensor_proto([96,112,53,32,5], shape=[5]))
    request.inputs['C25'].CopyFrom(tf.contrib.util.make_tensor_proto([0,0,0,0,1], shape=[5]))
    request.inputs['C26'].CopyFrom(tf.contrib.util.make_tensor_proto([0,0,0,0,47], shape=[5]))

    result = stub.Predict(request, 5.0)  # 5 secs timeout

    outputs_tensor_proto = result.outputs["logits"]
    shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
    outputs = tf.constant(list(outputs_tensor_proto.float_val), shape=shape)
    outputs = np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())

    print(f'logits:{outputs}')
    outputs_tensor_proto = result.outputs["pred"]
    shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
    outputs = tf.constant(list(outputs_tensor_proto.float_val), shape=shape)
    outputs = np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())
    print(f'pred:{outputs}')

if __name__ == "__main__":
    data = pd.read_csv('../criteo_sample.txt')
    df = data.head() ## for generate serving samples

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    feat_index_dict = {} 
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        feat_index_dict.update({feat:get_integer_mapping(lbe)})

    # save min max value for each dense feature 
    s_max,s_min = data[dense_features].max(axis=0),data[dense_features].min(axis=0)
    pd.concat([s_max, s_min],keys=['max','min'],axis=1).to_csv(f"max_min.txt",sep="\t")
        
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # save category features index for serving stage
    import json
    with open("feat_index_dict.json", 'w') as fo:
        json.dump(feat_index_dict, fo)
    

    # 2.count #unique features for each sparse field,and record dense feature field name
    dnn_feature_columns = []
    linear_feature_columns = []
    for i, feat in enumerate(sparse_features):
        dnn_feature_columns.append(tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(feat, data[feat].nunique()), 4))
        linear_feature_columns.append(tf.feature_column.categorical_column_with_identity(feat, data[feat].nunique()))
    for feat in dense_features:
        dnn_feature_columns.append(tf.feature_column.numeric_column(feat))
        linear_feature_columns.append(tf.feature_column.numeric_column(feat))

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2)
    # Not setting default value for continuous feature. filled with mean.
    train_model_input = input_fn_pandas(train,sparse_features+dense_features,'label')
    test_model_input = input_fn_pandas(test,sparse_features+dense_features,None)

    # 4.Define Model,train,predict and evaluate
    model = DeepFMEstimator(linear_feature_columns, dnn_feature_columns)
    model.train(train_model_input)
    pred_ans_iter = model.predict(test_model_input)
    pred_ans = list(map(lambda x:x['pred'],pred_ans_iter))
    #
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

    # 5.saved Model by build_raw_serving_input ,generate model in export_path
    def serving_input_receiver_fn():
        feature_map = {}
        for i in range(len(sparse_features)):
            feature_map[sparse_features[i]] = tf.placeholder(tf.int32,shape=(None, ),name='{}'.format(sparse_features[i]))
        for i in range(len(dense_features)):
            feature_map[dense_features[i]] = tf.placeholder(tf.float32,shape=(None, ),name='{}'.format(dense_features[i]))
        return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_map)
        
    model.export_savedmodel(export_dir_base='./serving_raw/',
                                serving_input_receiver_fn=serving_input_receiver_fn())

    
    # 6. run bash test serving service in local PC
    # export_path = './serving_raw/1593253204' (generated above)
    # run bash 
    # !saved_model_cli show --dir {export_path} --all
    # !saved_model_cli run --dir {export_path} --tag_set serve --signature_def "serving_default" --input_expr 'I1=[0.0,0.0,0.0,0.0,0.0];I2=[0.001332,0.0,0.000333,0.004664,0.000333];I3=[0.092362,0.00675,0.00071,0.000355,0.036945];I4=[0.0,0.402298,0.137931,0.045977,0.310344];I5=[0.034825,0.059628,0.003968,0.033185,0.003922];I6=[0.0,0.117284,0.077873,0.094967,0.067426];I7=[0.0,0.003322,0.019934,0.016611,0.013289];I8=[0.673468,0.714284,0.714284,0.081632,0.65306];I9=[0.0,0.154739,0.505803,0.028046,0.035783];I10=[0.0,0.0,0.0,0.0,0.0];I11=[0.0,0.03125,0.09375,0.0625,0.03125];I12=[0.0,0.0,0.0,0.0,0.0];I13=[0.0,0.343137,0.17647,0.039216,0.264706];C1=[0,11,0,0,0];C2=[4,1,18,45,11];C3=[96,98,39,7,59];C4=[146,98,52,117,77];C5=[1,1,3,1,1];C6=[4,6,4,0,5];C7=[163,179,140,164,18];C8=[1,0,2,1,1];C9=[1,1,1,0,1];C10=[72,89,93,20,45];C11=[117,58,31,61,171];C12=[127,97,122,104,162];C13=[157,79,16,36,96];C14=[7,7,7,1,4];C15=[127,72,129,43,36];C16=[126,26,97,43,121];C17=[8,7,8,8,8];C18=[66,52,49,37,14];C19=[0,0,0,0,5];C20=[0,0,0,0,3];C21=[3,47,25,156,9];C22=[0,0,0,0,0];C23=[1,7,6,0,0];C24=[96,112,53,32,5];C25=[0,0,0,0,1];C26=[0,0,0,0,47]'
    # # local pc import os
    # os.environ["MODEL_DIR"] = '/home/mi/openwork/sub/DeepCTR/examples/estimator_serving/serving_raw'
    # 
    # bash  
    # nohup tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=raw_export_deepfm_model --model_base_path=${MODEL_DIR} >server.log 2>&1

    # test_REST_serving()
    # test_RPC_serving()