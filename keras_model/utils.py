from keras.engine.topology import Layer
from keras.layers import Lambda,subtract,multiply,Dense,Activation,Dot,Reshape,Concatenate,RepeatVector
import keras.backend as K

def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

class FMLayer(Layer):
    def __init__(self, **kwargs):
        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(FMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, concated_embeds_value):
        """

        :param concated_embeds_value: None * field_size * embedding_size
        :return:
        """
        temp_a = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), )(concated_embeds_value)
        temp_b = multiply([concated_embeds_value, concated_embeds_value])
        temp_b = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(temp_b)
        cross_term = subtract([temp_a, temp_b])
        cross_term = Lambda(lambda x: 0.5 * K.sum(x, axis=2, keepdims=False))(cross_term)
        return cross_term


    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

class AFMLayer(Layer):
    def __init__(self,attention_factor=4, **kwargs):
        self.attention_factor = attention_factor
        super(AFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(AFMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, embeds_vec_list):
        """

        :param embeds_vec_list:  list of len(field_size) ,elem: embedding_size
        :return:
        """

        # TODO: biinteraction
        bi_interaction = []
        for i in range(len(embeds_vec_list)):
            for j in range(i + 1, len(embeds_vec_list)):
                prod_merged = multiply([embeds_vec_list[i], embeds_vec_list[j]])
                bi_interaction.append(prod_merged)
        bi_interaction_num = len(bi_interaction)
        bi_interaction = concatenate(bi_interaction,axis=1)
        attention_temp = Dense(self.attention_factor, activation='relu')(bi_interaction)
        attention_a = Dense(1, activation=None)(attention_temp)
        attention_weight = Activation(activation='softmax')(attention_a)
        attention_output = Dot(axes=1)([attention_weight, bi_interaction])
        afm_out = Reshape([1])(Dense(1, )(attention_output))
        return afm_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

def IdentityLayer(x):
    return Lambda(lambda x: x, output_shape=lambda s: s)(x)

def ActivationWeightedSumLayer(vec_list, target_vec,):
    """

    :param vec_list:  None * vec_num * vec_size
    :param target_vec:  None * vec_size
    :return:
    """
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    _,vec_nums,vec_size = K.int_shape(vec_list)
    repeat_target = RepeatVector(vec_nums)(target_vec)
    substract_vec = subtract([vec_list,repeat_target])
    concatenate_vec = Concatenate()([vec_list,substract_vec,repeat_target])
    attention_temp = Dense(vec_size, activation='relu')(concatenate_vec)
    attention_output = multiply([attention_temp, vec_list])
    ans  = Lambda(lambda x:K.sum(x,axis=1,keepdims=True))(attention_output)
    return ans

def ActivationSoftmaxWeightedSumLayer(vec_list, target_vec,):
    """

    :param vec_list:  None * vec_num * vec_size
    :param target_vec:  None * vec_size
    :return:
    """
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    _,vec_nums,vec_size = K.int_shape(vec_list)
    repeat_target = RepeatVector(vec_nums)(target_vec)
    substract_vec = subtract([vec_list,repeat_target])
    concatenate_vec = Concatenate()([vec_list,substract_vec,repeat_target])
    attention_temp = Dense(vec_size, activation='tanh')(concatenate_vec)
    attention_temp2 = Dense(1,activation='relu')(attention_temp)
    attention_score = Activation(softmax, )(attention_temp2)
    ans = Dot(axes=1)([attention_score, vec_list])
    return ans

def SoftmaxWeightedSumLayer(vec_list,):
    """

    :param vec_list:  None * vec_num * vec_size
    :return:
    """
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    _,vec_nums,vec_size = K.int_shape(vec_list)
    attention_temp = Dense(vec_size, activation='tanh')(vec_list)
    attention_temp2 = Dense(1,activation='relu')(attention_temp)
    attention_score = Activation(softmax, )(attention_temp2)
    ans = Dot(axes=1)([attention_score, vec_list])
    #print(attention_score,vec_list,ans)
    return ans

def WeightedSumLayer(vec_list, ):
    """

    :param vec_list:  None * vec_num * vec_size
    :param target_vec:  None * vec_size
    :return:
    """
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    _,vec_nums,vec_size = K.int_shape(vec_list)
    attention_temp2 = Dense(vec_size, activation='tanh')(vec_list)
    attention_temp = Dense(1, activation='relu')(attention_temp2)
    attention_output = multiply([attention_temp, vec_list])
    #print(attention_temp, attention_output)
    ans  = Lambda(lambda x:K.sum(x,axis=1,keepdims=True))(attention_output)
    return ans