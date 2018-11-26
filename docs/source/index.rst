.. DeepCTR documentation master file, created by
   sphinx-quickstart on Fri Nov 23 21:08:54 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepCTR's documentation!
===================================

DeepCTR is a **Easy-to-use** , **Modular** and **Extendible** package of deep-learning based CTR models along with lots of core components layer  which can be used to build your own custom model easily.You can use any complex model with ``model.fit()`` and ``model.predict()`` just like any other keras model.And the layers are compatible with tensorflow.

Through  ``pip install deepctr``  get the package and `Get Started! <./Quick-Start.html>`_

You can read the source code at https://github.com/shenweichen/DeepCTR

.. toctree::
   :maxdepth: 2
   :caption: Home:

   Quick-Start
   Features
   Demo

.. toctree::
   :maxdepth: 3
   :caption: API:

   Models API<Models-API>
   Layers API<deepctr.layers.rst>
   Activations API<deepctr.activations.rst>
   Sequence API<deepctr.sequence.rst>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`