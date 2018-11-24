Quick-Start
===========

Installation Guide
----------------------
Install deepctr package is through ``pip``.You must make sure that you have already installed tensorflow on your local machine: ::

    pip install deepctr


Getting started: 4 steps to DeepCTR
-----------------------------------------


Step 1: Import model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder,MinMaxScaler

    from deepctr import DeepFM

    data = pd.read_csv('./criteo_sample.txt')

    sparse_features  = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I'+str(i) for i in range(1,14)]
    target = ['label']

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0,)

    


Step 2: Simple preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Usually there are two simple way to encode the sparse feature for embedding

- Label Encoding: map the features to integer value from 0 ~ len(#unique) - 1
- Hash Encoding: map the features to a fix range,like 0 ~ 9999

And for dense features,they are usually  discretized to buckets,here we use normalization.

.. code-block:: python

    for feat in sparse_features:
        lbe = LabelEncoder()# or Hash
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0,1))
    data[dense_features] = mms.fit_transform(data[dense_features])



Step 3: Generate feature config dict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, for sparse features, we transform them into dense vectors by embedding techniques.
For continuous value features, we add a dummy index like LIBFM.
That is, all dense features under the same field share the same embedding vector.
In some implementations, the dense feature is concatened to the input embedding vectors of the deep network, you can modify the code yourself.


.. code-block:: python

    sparse_feature_dict = {feat: data[feat].nunique() for feat in sparse_features}
    dense_feature_list = dense_features


Step 4: Generate the training samples and train the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two rules here that we must follow

  - The sparse features are placed in front of the dense features.
  - The order of the feature we fit into the model must be consistent with the order of the feature dictionary iterations

.. code-block:: python

    # make sure the order is right
    model_input = [data[feat].values for feat in sparse_feature_dict] + [data[feat].values for feat in dense_feature_list]

    model = DeepFM({"sparse": sparse_feature_dict, "dense": dense_feature_list}, final_activation='sigmoid')
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
    history = model.fit(model_input, data[target].values,
                        batch_size=256, epochs=1, verbose=2, validation_split=0.2,)


You can check the full code `here <./Demo.html>`_








