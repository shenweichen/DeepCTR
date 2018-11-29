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

2. Does the models support multi-value input?

Now only the `DIN <Features.html#din-deep-interest-network>`_ model support multi-value input,you can use layers in `sequence <deepctr.sequence.html>`_ to build your own models!
And I will add the feature soon~