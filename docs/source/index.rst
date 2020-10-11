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

DeepCTR is a **Easy-to-use** , **Modular** and **Extendible** package of deep-learning based CTR models along with lots of core components layer  which can be used to easily build custom models.You can use any complex model with ``model.fit()`` and ``model.predict()``.

- Provide ``tf.keras.Model`` like interface for **quick experiment**. `example <https://deepctr-doc.readthedocs.io/en/latest/Quick-Start.html#getting-started-4-steps-to-deepctr>`_
- Provide  ``tensorflow estimator`` interface for **large scale data** and **distributed training**. `example <https://deepctr-doc.readthedocs.io/en/latest/Quick-Start.html#getting-started-4-steps-to-deepctr-estimator-with-tfrecord>`_
- It is compatible with both ``tf 1.x``  and ``tf 2.x``.

Let's `Get Started! <./Quick-Start.html>`_ (`Chinese Introduction <https://zhuanlan.zhihu.com/p/53231955>`_)

You can read the latest code and related projects

- DeepCTR: https://github.com/shenweichen/DeepCTR
- DeepMatch: https://github.com/shenweichen/DeepMatch
- DeepCTR-Torch: https://github.com/shenweichen/DeepCTR-Torch

News
-----

10/11/2020 : Refactor ``DNN`` Layer. `Changelog <https://github.com/shenweichen/DeepCTR/releases/tag/v0.8.2>`_

09/12/2020 : Improve the reproducibility & fix some bugs. `Changelog <https://github.com/shenweichen/DeepCTR/releases/tag/v0.8.1>`_

06/27/2020 : Support ``Tensorflow Estimator`` for large scale data and distributed training.Support different initializers for different embedding weights and loading pretrained embeddings.Add new model ``FwFM``. `Changelog <https://github.com/shenweichen/DeepCTR/releases/tag/v0.8.0>`_

DisscussionGroup 交流群
-----------------------

公众号：**浅梦的学习笔记**  wechat ID: **deepctrbot**

.. image:: ../pics/code.png

Cooperative promotion 合作推广
------------------------------

For more information about the recommendation system, such as **feature engineering, user profile, matching, ranking and multi-objective optimization, online learning and real-time computing, and more cutting-edge technologies and practical projects** :

更多关于推荐系统的内容，如 **特征工程，用户画像，召回，排序和多目标优化，在线学习与实时计算以及更多前沿技术和实战项目** 等可参考：

- `推荐系统实战 <https://www.julyedu.com/course/getDetail/181?ccode=5ee751d37278c>`_
- `互联网计算广告实战 <https://www.julyedu.com/course/getDetail/158?ccode=5ee751d37278c>`_



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
   Estimators<Estimators>
   Layers<Layers>




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`