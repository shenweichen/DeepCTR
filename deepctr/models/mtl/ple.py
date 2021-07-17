"""
Author:
    Mincai Lai, laimc@shanghaitech.edu.cn

Reference:
    [1] Tang H, Liu J, Zhao M, et al. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations[C]//Fourteenth ACM Conference on Recommender Systems. 2020.(https://dl.acm.org/doi/10.1145/3383313.3412236)
"""

import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import combined_dnn_input, reduce_sum 

def PLE(dnn_feature_columns, num_tasks, task_types, task_names, num_levels=2, num_experts_specific=8, num_experts_shared=4,
        expert_dnn_units=[64,64],  gate_dnn_units=None, tower_dnn_units_lists=[[16,16],[16,16]],
        l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False):
    """Instantiates the multi level of Customized Gate Control of Progressive Layered Extraction architecture.
    
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks
    
    :param num_levels: integer, number of CGC levels.
    :param num_experts_specific: integer, number of task-specific experts.
    :param num_experts_shared: integer, number of task-shared experts.

    :param expert_dnn_units: list, list of positive integer, its length must be greater than 1, the layer number and units in each layer of expert DNN.
    :param gate_dnn_units: list, list of positive integer or None, the layer number and units in each layer of gate DNN, default value is None. e.g.[8, 8].
    :param tower_dnn_units_lists: list, list of positive integer list, its length must be euqal to num_tasks, the layer number and units in each layer of task-specific DNN.
    
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN.
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN.
    :return: a Keras model instance.
    """
    
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")
        
    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))
            
    if num_tasks != len(tower_dnn_units_lists):
        raise ValueError("the length of tower_dnn_units_lists must be euqal to num_tasks")

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
 
    #single Extraction Layer
    def cgc_net(inputs, level_name, is_last=False):
        #inputs: [task1, task2, ... taskn, shared task]
        expert_outputs = []
        #build task-specific expert layer
        for i in range(num_tasks):
            for j in range(num_experts_specific):
                expert_network = DNN(expert_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name=level_name+'task_'+task_names[i]+'_expert_specific_'+str(j))(inputs[i])
                expert_outputs.append(expert_network)

        #build task-shared expert layer
        for i in range(num_experts_shared):
            expert_network = DNN(expert_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name=level_name+'expert_shared_'+str(i))(inputs[-1]) 
            expert_outputs.append(expert_network)

        #task_specific gate (count = num_tasks)
        cgc_outs = []
        for i in range(num_tasks): 
            #concat task-specific expert and task-shared expert
            cur_expert_num = num_experts_specific + num_experts_shared
            cur_experts = expert_outputs[i * num_experts_specific:(i + 1) * num_experts_specific] + expert_outputs[-int(num_experts_shared):] #task_specific + task_shared
            
            expert_concat = tf.keras.layers.concatenate(cur_experts, axis=1, name=level_name+'expert_concat_specific_'+task_names[i])
            expert_concat = tf.keras.layers.Reshape([cur_expert_num, expert_dnn_units[-1]], name=level_name+'expert_reshape_specific_'+task_names[i])(expert_concat)

            #build gate layers
            if gate_dnn_units!=None:
                gate_network = DNN(gate_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name=level_name+'gate_specific_'+task_names[i])(inputs[i]) #gate[i] for task input[i]
                gate_input = gate_network
            else:  #in origin paper, gate is one Dense layer with softmax.
                gate_input = dnn_input        
            gate_out = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax', name=level_name+'gate_softmax_specific_'+task_names[i])(gate_input)
            gate_out = tf.tile(tf.expand_dims(gate_out, axis=-1), [1, 1, expert_dnn_units[-1]]) 

            #gate multiply the expert
            gate_mul_expert = tf.keras.layers.Multiply(name=level_name+'gate_mul_expert_specific_'+task_names[i])([expert_concat, gate_out]) 
            gate_mul_expert = reduce_sum(gate_mul_expert, axis=1, keep_dims=True) #sum pooling in the expert ndim
            cgc_outs.append(gate_mul_expert)
        
        #task_shared gate, if the level not in last, add one shared gate
        if not is_last:
            cur_expert_num = num_tasks * num_experts_specific + num_experts_shared
            cur_experts = expert_outputs #all the expert include task-specific expert and task-shared expert
            
            expert_concat = tf.keras.layers.concatenate(cur_experts, axis=1, name=level_name+'expert_concat_shared_'+task_names[i])
            expert_concat = tf.keras.layers.Reshape([cur_expert_num, expert_dnn_units[-1]], name=level_name+'expert_reshape_shared_'+task_names[i])(expert_concat)
            
            #build gate layers
            if gate_dnn_units!=None:
                gate_network = DNN(gate_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name=level_name+'gate_shared_'+str(i))(inputs[-1])  #gate for shared task input
                gate_input = gate_network
            else:  #in origin paper, gate is one Dense layer with softmax.
                gate_input = dnn_input  
                
            gate_out = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax', name=level_name+'gate_softmax_shared_'+str(i))(gate_input)
            gate_out = tf.tile(tf.expand_dims(gate_out, axis=-1), [1, 1, expert_dnn_units[-1]]) 

            #gate multiply the expert
            gate_mul_expert = tf.keras.layers.Multiply(name=level_name+'gate_mul_expert_shared_'+task_names[i])([expert_concat, gate_out]) 
            gate_mul_expert = reduce_sum(gate_mul_expert, axis=1, keep_dims=True) #sum pooling in the expert ndim
            cgc_outs.append(gate_mul_expert)
        return cgc_outs
    
    #build Progressive Layered Extraction
    ple_inputs = [dnn_input]*(num_tasks+1) #[task1, task2, ... taskn, shared task]
    ple_outputs = []
    for i in range(num_levels):
        if i == num_levels-1: #the last level
            ple_outputs = cgc_net(inputs=ple_inputs, level_name='level_'+str(i)+'_', is_last=True)
            break
        else:
            ple_outputs = cgc_net(inputs=ple_inputs, level_name='level_'+str(i)+'_', is_last=False)
            ple_inputs = ple_outputs
            
    task_outs = []
    for task_type, task_name, tower_dnn, ple_out in zip(task_types, task_names, tower_dnn_units_lists, ple_outputs):
        #build tower layer
        tower_output = DNN(tower_dnn, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='tower_'+task_name)(ple_out)
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit) 
        task_outs.append(output)
        
    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outs)
    return model