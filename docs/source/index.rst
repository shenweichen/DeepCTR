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


DeepCTR is a **Easy-to-use** , **Modular** and **Extendible** package of deep-learning based CTR models along with lots of core components layer  which can be used to build your own custom model easily.It is compatible with **tensorflow 1.4+ and 2.0+**.You can use any complex model with ``model.fit()`` and ``model.predict()``.

Let's `Get Started! <./Quick-Start.html>`_ (`Chinese Introduction <https://zhuanlan.zhihu.com/p/53231955>`_)

You can read the latest code at https://github.com/shenweichen/DeepCTR

News
-----
08/02/2019 : Now DeepCTR is compatible with tensorflow `1.14` and `2.0.0`. `Changelog <https://github.com/shenweichen/DeepCTR/releases/tag/v0.6.0>`_

07/21/2019 : Refactor Linear Layer. `Changelog <https://github.com/shenweichen/DeepCTR/releases/tag/v0.5.2>`_

07/10/2019 : Add `FiBiNEt <./Features.html#fibinet-feature-importance-and-bilinear-feature-interaction-network>`_ . `Changelog <https://github.com/shenweichen/DeepCTR/releases/tag/v0.5.1>`_

.. toctree::
   :maxdepth: 2
   :caption: Home:

   Quick-Start<Quick-Start.md>
   Features<Features.md>
   Examples<Examples.md>
   FAQ<FAQ.md>
   History<History.md>

.. toctree::
   :maxdepth: 3
   :caption: API:

   Models<Models>
   Layers<Layers>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`