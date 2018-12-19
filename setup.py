import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'tensorflow>=1.4.0,!=1.5.0,!=1.7.*,!=1.8.0'
]

setuptools.setup(
    name="deepctr",
    version="0.1.6",
    author="Weichen Shen",
    author_email="wcshen1994@163.com",
    description="DeepCTR is a Easy-to-use,Modular and Extendible package of deep-learning based CTR(Click Through Rate) prediction models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shenweichen/deepctr",
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires='>=3.4.6',
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        "tf": ['tensorflow>=1.4.0,!=1.5.0,!=1.7.*,!=1.8.0'],
        "tf_gpu": ['tensorflow-gpu>=1.4.0,!=1.5.0,!=1.7.*,!=1.8.0'],
    },
    entry_points={
    },
    classifiers=(
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ),
    license="MIT license",
    keywords=['ctr', 'click through rate', 'deep learning', 'tensorflow']
)
