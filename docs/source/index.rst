.. DeepCTR documentation master file, created by
   sphinx-quickstart on Fri Nov 23 21:08:54 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepCTR's documentation!
===================================

|Downloads|_ |Stars|_ |Forks|_ |PyPii|_ |Issues|_ |Chat|_

.. |Downloads| image:: https://pepy.tech/badge/deepctr
.. _Downloads: https://pepy.tech/project/deepctr

.. |Stars| image:: https://img.shields.io/github/stars/shenweichen/deepctr.svg
.. _Stars: https://github.com/shenweichen/DeepCTR

.. |Forks| image:: https://img.shields.io/github/forks/shenweichen/deepctr.svg
.. _Forks: https://github.com/shenweichen/DeepCTR/fork

.. |PyPii| image:: https://img.shields.io/pypi/v/deepctr.svg
.. _PyPii: https://pypi.org/project/deepctr

.. |Issues| image:: https://img.shields.io/github/issues/shenweichen/deepctr.svg
.. _Issues: https://github.com/shenweichen/deepctr/issues

.. |Chat| image:: https://img.shields.io/badge/chat-wechat-brightgreen?style=flat
.. _Chat: ./#disscussiongroup

DeepCTR is a **Easy-to-use** , **Modular** and **Extendible** package of deep-learning based CTR models along with lots of core components layer  which can be used to easily build custom models.It is compatible with **tensorflow 1.4+ and 2.0+**.You can use any complex model with ``model.fit()`` and ``model.predict()``.

Let's `Get Started! <./Quick-Start.html>`_ (`Chinese Introduction <https://zhuanlan.zhihu.com/p/53231955>`_)

You can read the latest code at https://github.com/shenweichen/DeepCTR

News
-----
11/24/2019 : Refactor `feature columns <./Features.html#feature-columns>`_ . Different features can use different ``embedding_dim`` and  group-wise interaction is available by setting ``group_name``. `Changelog <https://github.com/shenweichen/DeepCTR/releases/tag/v0.7.0>`_

11/06/2019 : Add ``WeightedSequenceLayer`` and support `weighted sequence feature input <./Examples.html#multi-value-input-movielens>`_. `Changelog <https://github.com/shenweichen/DeepCTR/releases/tag/v0.6.3>`_

10/03/2019 : Simplify the input logic(`examples <./Examples.html#classification-criteo>`_). `Changelog <https://github.com/shenweichen/DeepCTR/releases/tag/v0.6.2>`_

DisscussionGroup
-----------------------

公众号：**浅梦的学习笔记**  wechat ID: **deepctrbot**

.. image:: ../pics/weichennote.png

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