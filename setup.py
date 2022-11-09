import sys

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'requests',
    'h5py==3.7.0; python_version>="3.9"',
    'h5py==2.10.0; python_version<"3.9"'
]

setuptools.setup(
    name="deepctr",
    version="0.9.3",
    author="Weichen Shen",
    author_email="weichenswc@163.com",
    description="Easy-to-use,Modular and Extendible package of deep learning based CTR(Click Through Rate) prediction models with tensorflow 1.x and 2.x .",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shenweichen/deepctr",
    download_url='https://github.com/shenweichen/deepctr/tags',
    packages=setuptools.find_packages(
        exclude=["tests", "tests.models", "tests.layers"]),
    python_requires=">=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*",  # '>=3.4',  # 3.4.6
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        "cpu": ["tensorflow>=1.4.0,!=1.7.*,!=1.8.*"],
        "gpu": ["tensorflow-gpu>=1.4.0,!=1.7.*,!=1.8.*"],
    },
    entry_points={
    },
    classifiers=(
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ),
    license="Apache-2.0",
    keywords=['ctr', 'click through rate',
              'deep learning', 'tensorflow', 'tensor', 'keras'],
)
