import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepctr",
    version="0.1.6",
    author="Weichen Shen",
    author_email="wcshen1994@163.com",
    description="DeepCTR is a Easy-to-use,Modular and Extendible package of deep-learning based CTR models ,including serval DNN-based CTR models and lots of core components layer of the models which can be used to build your own custom model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shenweichen/deepctr",
    packages=setuptools.find_packages(),
    install_requires=[],
    extras_require={
        "tf": ["tensorflow>=1.4.0,<1.7.0"],
        "tf_gpu": ["tensorflow-gpu>=1.4.0,<1.7.0"],
    },
    entry_points={
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
