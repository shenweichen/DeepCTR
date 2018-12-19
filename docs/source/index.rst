.. DeepCTR documentation master file, created by
   sphinx-quickstart on Fri Nov 23 21:08:54 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepCTR's documentation!
===================================

|PyPi|_ |Downloads|_ |Stars|_ |Forks|_ |Activaty|_

.. |PyPi| image:: https://img.shields.io/pypi/v/deepctr.svg
.. _PyPi: https://pypi.org/project/deepctr/

.. |Downloads| image:: https://pepy.tech/badge/deepctr
.. _Downloads: https://pepy.tech/project/deepctr

.. |Stars| image:: https://img.shields.io/github/stars/shenweichen/deepctr.svg
.. _Stars: https://github.com/shenweichen/DeepCTR

.. |Forks| image:: https://img.shields.io/github/forks/shenweichen/deepctr.svg
.. _Forks: https://github.com/shenweichen/DeepCTR

.. |Activaty| image:: https://img.shields.io/github/last-commit/shenweichen/deepctr.svg
.. _Activaty: https://github.com/shenweichen/DeepCTR

DeepCTR is a **Easy-to-use** , **Modular** and **Extendible** package of deep-learning based CTR models along with lots of core components layer  which can be used to build your own custom model easily.You can use any complex model with ``model.fit()`` and ``model.predict()`` just like any other keras model.And the layers are compatible with tensorflow.

Through  ``pip install deepctr``  get the package and `Get Started! <./Quick-Start.html>`_

You can read the latest code at https://github.com/shenweichen/DeepCTR

.. toctree::
   :maxdepth: 2
   :caption: Home:

   Quick-Start
   Features
   Demo
   FAQ

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