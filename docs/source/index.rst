.. DeepCTR documentation master file, created by
   sphinx-quickstart on Fri Nov 23 21:08:54 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepCTR's documentation!
===================================

DeepCTR is a **Easy-to-use** , **Modular** and **Extendible** package of deep-learning based based CTR models ,including serval DNN-based CTR models and lots of core components layer of the models which can be used to build your own custom model.
The goal is to make it possible for everyone to use complex deep learning-based models with ``model.fit()`` and ``model.predict()`` .

Through ``pip install deepctr`` get the package and `Get Started! <./Quick-Start.html>`_

You can find source code at https://github.com/shenweichen/DeepCTR

.. toctree::
   :maxdepth: 2
   :caption: Home:

   Quick-Start
   Features
   Demo

.. toctree::
   :maxdepth: 3
   :caption: APIs:

   Models API<Models-API>
   Layers API<deepctr.layers.rst>
   Activations API<deepctr.activations.rst>
   Sequence API<deepctr.sequence.rst>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`