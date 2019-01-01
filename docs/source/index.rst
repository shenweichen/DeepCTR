.. DeepCTR documentation master file, created by
   sphinx-quickstart on Fri Nov 23 21:08:54 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepCTR's documentation!
===================================

|Downloads|_ |Stars|_ |Forks|_ |PyPi|_ |Issues|_ |Activity|_

.. |Downloads| image:: https://pepy.tech/badge/deepctr
.. _Downloads: https://pepy.tech/project/deepctr

.. |Stars| image:: https://img.shields.io/github/stars/shenweichen/deepctr.svg
.. _Stars: https://github.com/shenweichen/DeepCTR

.. |Forks| image:: https://img.shields.io/github/forks/shenweichen/deepctr.svg
.. _Forks: https://github.com/shenweichen/DeepCTR/fork

.. |PyPi| image:: https://img.shields.io/pypi/v/deepctr.svg
.. _PyPi: https://pypi.org/project/deepctr/

.. |Issues| image:: https://img.shields.io/github/issues/shenweichen/deepctr.svg
.. _Issues: https://github.com/shenweichen/deepctr/issues

.. |Activity| image:: https://img.shields.io/github/last-commit/shenweichen/deepctr.svg
.. _Activity: https://github.com/shenweichen/DeepCTR


DeepCTR is a **Easy-to-use** , **Modular** and **Extendible** package of deep-learning based CTR models along with lots of core components layer  which can be used to build your own custom model easily.You can use any complex model with ``model.fit()`` and ``model.predict()``.And the layers are compatible with tensorflow.

Through  ``pip install deepctr``  get the package and `Get Started! <./Quick-Start.html>`_

You can read the latest code at https://github.com/shenweichen/DeepCTR

News
-----
01/01/2019 : Add `sequence(multi-value) input support for AFM,AutoInt,DCN,DeepFM,FNN,NFM,PNN,xDeepFM <./Features.html#autoint-automatic-feature-interactiont) models>`_.`Changelog <https://github.com/shenweichen/DeepCTR/releases/tag/v0.2.2>`_

12/27/2018 : Add `AutoInt <./Features.html#autoint-automatic-feature-interaction>`_ . `Changelog <https://github.com/shenweichen/DeepCTR/releases/tag/v0.2.1>`_

12/22/2018 : Add `xDeepFM <./Features.html#xdeepfm>`_ and automatic check for new version. `Changelog <https://github.com/shenweichen/DeepCTR/releases/tag/v0.2.0>`_

12/19/2018 : DeepCTR is compatible with tensorflow from ``1.4-1.12`` except for ``1.7`` and ``1.8``. `Changelog <https://github.com/shenweichen/DeepCTR/releases/tag/v0.1.6>`_

.. toctree::
   :maxdepth: 2
   :caption: Home:

   Quick-Start
   Features
   Examples<Examples.md>
   FAQ
   History<History.md>

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