# -*- coding:utf-8 -*-
"""
@author: shenweichen,wcshen1994@163.com
A keras implementation of MLR
Reference:
[1] Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction(https://arxiv.org/abs/1704.05194)
"""
from keras.layers import Input, Dense, Embedding, Concatenate, Activation,  Reshape,  add, dot
from keras.models import Model
from keras.initializers import TruncatedNormal
from keras.regularizers import l2


class MLR():
    def __init__(self, region_feature_dim,base_feature_dim={"sparse":{},"dense":[]},region_num=4,
                 l2_reg_linear=0.00002,
                 init_std=0.0001, activation = 'sigmoid',seed=1024,
                 checkpoint_path=None,bias_feature_dim={"sparse":{},"dense":[]},):
      """
      if not assign base_feature_dim,it will equal to region_feature_dim
      :param region_feature_dim: dict
      :param base_feature_dim: dict
      :param region_num: int
      :param l2_reg_linear:
      :param init_std:
      :param seed:
      :param checkpoint_path:
      :param bias_feature_dim: dict
      """

      if region_num <= 1:
        raise ValueError("region_num must > 1")
      if not isinstance(region_feature_dim,
                        dict) or "sparse" not in region_feature_dim or "dense" not in region_feature_dim:
          raise ValueError(
              "feature_dim must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_5',]}")

      self.same_flag = False
      if base_feature_dim == {"sparse":{},"dense":[]}:
          base_feature_dim = region_feature_dim
          self.same_flag = True

      self.region_cate_feature_dim = region_feature_dim["sparse"]
      self.region_continuous_feature_dim = region_feature_dim["dense"]
      self.base_cate_feature_dim = base_feature_dim["sparse"]
      self.base_continuous_feature_dim = base_feature_dim["dense"]
      self.bias_cate_feature_dim = bias_feature_dim["sparse"]
      self.bias_continuous_feature_dim = bias_feature_dim["dense"]
      self.region_num = region_num
      self.final_activation = activation
      self.l2_reg_linear = 0#l2_reg_linear
      self.init_std = init_std
      self.seed = seed
      self.checkpoint_path = checkpoint_path
      self.model = self.create_model()

    def get_model(self, ):
        return self.model

    def create_model(self,):
        region_cate_input, region_continuous_input, base_cate_input, base_continuous_input, bias_cate_input, bias_continuous_input = self.get_input( )
        region_embeddings, base_embeddings,bias_embedding= self.create_cate_embedding()

        if self.same_flag:
            base_continuous_input_ = region_continuous_input
            base_cate_input_ = region_cate_input
        else:
            base_continuous_input_ = base_continuous_input
            base_cate_input_ = base_cate_input



        if len(self.region_continuous_feature_dim) > 1:
            region_continous_logits = [Dense(1, )(Concatenate()(region_continuous_input)) for _ in
                                       range(self.region_num)]
        elif len(self.region_continuous_feature_dim) == 1:
            region_continous_logits = [Dense(1, )(region_continuous_input[0]) for _ in
                                       range(self.region_num)]

        if len(self.base_continuous_feature_dim) > 1:
            base_continous_logits = [Dense(1, )(Concatenate()(base_continuous_input_))for _ in
                                 range(self.region_num)]
        elif len(self.base_continuous_feature_dim)==1:
            base_continous_logits = [Dense(1, )(base_continuous_input_[0])for _ in
                                 range(self.region_num)]


        if len(self.region_continuous_feature_dim) > 0 and len(self.region_cate_feature_dim)==0:
            region_logits = Concatenate()(region_continous_logits)
            base_logits = base_continous_logits
        elif len(self.region_continuous_feature_dim) == 0 and len(self.region_cate_feature_dim) >0:
            region_cate_logits = [
                add([region_embeddings[j][i](region_cate_input[i]) for i in range(len(self.region_cate_feature_dim))])
                for j in range(self.region_num)]
            base_cate_logits = [add(
                [base_embeddings[j][i](base_cate_input_[i]) for i in range(len(self.base_cate_feature_dim))])
                for j in range(self.region_num)]
            region_logits = Concatenate()(region_cate_logits)
            base_logits = base_cate_logits
        else:
            region_cate_logits = [
                add([region_embeddings[j][i](region_cate_input[i]) for i in range(len(self.region_cate_feature_dim))])
                for j in range(self.region_num)]
            base_cate_logits = [add(
                [base_embeddings[j][i](base_cate_input_[i]) for i in range(len(self.base_cate_feature_dim))])
                for j in range(self.region_num)]
            region_logits =Concatenate()([add([region_cate_logits[i],region_continous_logits[i]]) for i in range(self.region_num)])
            base_logits = [add([base_cate_logits[i],base_continous_logits[i]]) for i in range(self.region_num)]
        region_weights = Activation("softmax")(region_logits)#Dense(self.region_num, activation='softmax')(final_logit)
        learner_score =  Concatenate()(
            [Activation(self.final_activation, name='learner' + str(i))(base_logits[i]) for i in range(self.region_num)])
        final_logit = dot([region_weights,learner_score], axes=-1)

        if len(self.bias_continuous_feature_dim) + len(self.bias_cate_feature_dim) >0:
            if len(self.bias_continuous_feature_dim) > 1:
                bias_continous_logits =Dense(1, )(Concatenate()(bias_continuous_input))
            else:
                bias_continous_logits = Dense(1, )(bias_continuous_input[0])
            bias_cate_logits = add([bias_embedding[i](bias_cate_input[i]) for i, feat in enumerate(self.bias_cate_feature_dim)])
            if len(self.bias_continuous_feature_dim) > 0 and len(self.bias_cate_feature_dim) == 0:
                bias_logits = bias_continous_logits
            elif len(self.bias_continuous_feature_dim) == 0 and len(self.bias_cate_feature_dim) > 0:
                bias_logits = bias_cate_logits
            else:
                bias_logits = add([bias_continous_logits,bias_cate_logits])

            bias_prob = Activation('sigmoid')(bias_logits)
            final_logit = dot([final_logit,bias_prob],axes=-1)

        output = Reshape([1])(final_logit)
        model = Model(inputs=region_cate_input +region_continuous_input+base_cate_input+base_continuous_input+bias_cate_input+bias_continuous_input, outputs=output)
        return model

    def get_input(self, ):
        region_cate_input = [Input(shape=(1,), name='region_cate_' + str(i)+"-"+feat) for i,feat in enumerate(self.region_cate_feature_dim)]
        region_continuous_input = [Input(shape=(1,), name='region_continuous_' + str(i)+"-"+feat) for i,feat in enumerate(self.region_continuous_feature_dim)]
        if self.same_flag == True:
            base_cate_input = []
            base_continuous_input = []
        else:
            base_cate_input = [Input(shape=(1,), name='base_cate_' + str(i) + "-" + feat) for i, feat in
                                 enumerate(self.base_cate_feature_dim)]
            base_continuous_input = [Input(shape=(1,), name='base_continuous_' + str(i) + "-" + feat) for i, feat in
                                       enumerate(self.base_continuous_feature_dim)]

        bias_cate_input = [Input(shape=(1,), name='bias_cate_' + str(i) + "-" + feat) for i, feat in
                             enumerate(self.bias_cate_feature_dim)]
        bias_continuous_input = [Input(shape=(1,), name='bias_continuous_' + str(i) + "-" + feat) for i, feat in
                                   enumerate(self.bias_continuous_feature_dim)]
        return  region_cate_input,region_continuous_input,base_cate_input,base_continuous_input,bias_cate_input,bias_continuous_input

    def create_cate_embedding(self, ):

        region_embeddings = [[Embedding(self.region_cate_feature_dim[feat], 1, embeddings_initializer=TruncatedNormal(stddev=self.init_std,seed=self.seed+j) \
                                      , embeddings_regularizer=l2(self.l2_reg_linear),
                                      name='embed_region' + str(j)+'_' + str(i)) for
                            i,feat in enumerate(self.region_cate_feature_dim)] for j in range(self.region_num)]
        base_embeddings = [[Embedding(self.base_cate_feature_dim[feat], 1,
                                        embeddings_initializer=TruncatedNormal(stddev=self.init_std, seed=self.seed + j) \
                                        , embeddings_regularizer=l2(self.l2_reg_linear),
                                        name='embed_base' + str(j) + '_' + str(i)) for
                              i, feat in enumerate(self.base_cate_feature_dim)] for j in range(self.region_num)]
        bias_embedding = [Embedding(self.bias_cate_feature_dim[feat], 1, embeddings_initializer=TruncatedNormal(stddev=self.init_std,seed=self.seed) \
                                      , embeddings_regularizer=l2(self.l2_reg_linear),
                                      name='embed_bias'  +'_' + str(i)) for
                            i,feat in enumerate(self.bias_cate_feature_dim)]

        return region_embeddings, base_embeddings,bias_embedding


if __name__ == "__main__":
    model = MLR({"sparse":{"field1":4,"field2":4},"dense":["as","ab"]},{"sparse":{"field3":4,"field4":4},"dense":["as"]},bias_feature_dim={"sparse":{"field5":4,"field6":3},"dense":["pos",]}).model
    model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy'])
    print("MLR compile done")