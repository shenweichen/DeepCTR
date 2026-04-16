import sys

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'requests',
    'h5py>=3.7.0; python_version>="3.9"',
    'h5py==2.10.0; python_version<"3.9"'
]

setuptools.setup(
    name="deepctr",
    version="0.9.4",
    author="Weichen Shen",
    author_email="weichenswc@163.com",
    description="Easy-to-use,Modular and Extendible package of deep learning based CTR(Click Through Rate) prediction models with tensorflow 1.x and 2.x .",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shenweichen/deepctr",
    download_url='https://github.com/shenweichen/deepctr/tags',
    packages=setuptools.find_packages(
        exclude=["tests", "tests.models", "tests.layers"]),
    python_requires=">=3.7",
    install_requires=REQUIRED_PACKAGES,
    extras_require={
    },
    entry_points={
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license="Apache-2.0",
    keywords=['ctr', 'click through rate',
              'deep learning', 'tensorflow', 'tensor', 'keras'],
)
