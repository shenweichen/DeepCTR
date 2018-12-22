# DeepCTR

[![Python Versions](https://img.shields.io/pypi/pyversions/deepctr.svg)](https://pypi.org/project/deepctr)
[![Downloads](https://pepy.tech/badge/deepctr)](https://pepy.tech/project/deepctr)
[![PyPI Version](https://img.shields.io/pypi/v/deepctr.svg)](https://pypi.org/project/deepctr)
[![GitHub Issues](https://img.shields.io/github/issues/shenweichen/deepctr.svg
)](https://github.com/shenweichen/deepctr/issues)
[![Activity](https://img.shields.io/github/last-commit/shenweichen/deepctr.svg)](https://github.com/shenweichen/DeepCTR/commits/master)


[![Documentation Status](https://readthedocs.org/projects/deepctr-doc/badge/?version=latest)](https://deepctr-doc.readthedocs.io/)
[![Build Status](https://travis-ci.org/shenweichen/DeepCTR.svg?branch=master)](https://travis-ci.org/shenweichen/DeepCTR)
[![Coverage Status](https://coveralls.io/repos/github/shenweichen/DeepCTR/badge.svg?branch=master)](https://coveralls.io/github/shenweichen/DeepCTR?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d4099734dc0e4bab91d332ead8c0bdd0)](https://www.codacy.com/app/wcshen1994/DeepCTR?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=shenweichen/DeepCTR&amp;utm_campaign=Badge_Grade)
[![License](https://img.shields.io/github/license/shenweichen/deepctr.svg)](https://github.com/shenweichen/deepctr/blob/master/LICENSE)

DeepCTR is a **Easy-to-use**,**Modular** and **Extendible** package of deep-learning based CTR models along with lots of core components layer  which can be used to build your own custom model easily.You can use any complex model with `model.fit()`and`model.predict()` just like any other keras model.And the layers are compatible with tensorflow.

Through  `pip install deepctr`  get the package and [**Get Started!**](https://deepctr-doc.readthedocs.io/en/latest/Quick-Start.html)


## Models List

|Model|Paper|
|:--:|:--|
|Factorization-supported Neural Network|[ECIR 2016][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)|
|Product-based Neural Network|[ICDM 2016][Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)|
|Wide & Deep|[arxiv 2016][Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)|
|DeepFM|[IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)|
|Piece-wise Linear Model|[arxiv 2017][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)|
|Deep & Cross Network|[ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)|
|Attentional Factorization Machine|[IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435)|
|Neural Factorization Machine|[SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)|
|Deep Interest Network|[KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)|
