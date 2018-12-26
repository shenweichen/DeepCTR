FAQ
==========
1. How to save or load weights/models?

To save/load weights,you can write codes just like any other keras models.

.. code-block:: python

    model = DeepFM()
    model.save_weights('DeepFM_w.h5')
    model.load_weights('DeepFM_w.h5')


To save/load models,just a little different.

.. code-block:: python

    from tensorflow.python.keras.models import  save_model,load_model
    model = DeepFM()
    save_model(model, 'DeepFM.h5')# save_model, same as before

    from deepctr.utils import custom_objects
    model = load_model('DeepFM.h5',custom_objects)# load_model,just add a parameter

2. How to set learning rate and use earlystopping?

You can use any models in DeepCTR like a keras model object.
Here is a example of how to set learning rate and earlystopping:

.. code-block:: python

    import deepctr
    from tensorflow.python.keras.optimizers import Adam,Adagrad
    from tensorflow.python.keras.callbacks import EarlyStopping

    model = deepctr.models.DeepFM({"sparse": sparse_feature_dict, "dense": dense_feature_list})
    model.compile(Adagrad('0.0808'),'binary_crossentropy',metrics=['binary_crossentropy'])

    es = EarlyStopping(monitor='val_binary_crossentropy')
    history = model.fit(model_input, data[target].values,batch_size=256, epochs=10, verbose=2, validation_split=0.2,callbacks=[es] )


3. How can I get the attentional weights of feature interactions in AFM?

First,make sure that you have install the latest version of deepctr.

Then,use the following code,the ``attentional_weights[:,i,0]`` is the ``feature_interactions[i]``'s attentional weight of all samples.

.. code-block:: python

    import itertools
    import deepctr
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.layers import Lambda

    feature_dim_dict = {"sparse": sparse_feature_dict, "dense": dense_feature_list}
    model = deepctr.models.AFM(feature_dim_dict)
    model.fit(model_input,target)

    afmlayer = model.layers[-3]
    afm_weight_model = Model(model.input,outputs=Lambda(lambda x:afmlayer.normalized_att_score)(model.input))
    attentional_weights = afm_weight_model.predict(model_input,batch_size=4096)
    feature_interactions = list(itertools.combinations(list(feature_dim_dict['sparse'].keys()) + feature_dim_dict['dense'] ,2))



4. Does the models support multi-value input?

Now only the `DIN <Features.html#din-deep-interest-network>`_ model support multi-value input,you can use layers in `sequence <deepctr.sequence.html>`_ to build your own models!
And I will add the feature soon~