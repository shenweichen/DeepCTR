from .utils import check_version

__version__ = '0.9.3'


def check_latest_version():
    """Check whether a newer DeepCTR version is available on PyPI."""
    check_version(__version__)
