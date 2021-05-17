# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com

"""

import json
import logging
from threading import Thread

import requests

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse


def check_version(version):
    """Return version of package on pypi.python.org using json."""

    def check(version):
        try:
            url_pattern = 'https://pypi.python.org/pypi/deepctr/json'
            req = requests.get(url_pattern)
            latest_version = parse('0')
            version = parse(version)
            if req.status_code == requests.codes.ok:
                j = json.loads(req.text.encode('utf-8'))
                releases = j.get('releases', [])
                for release in releases:
                    ver = parse(release)
                    if ver.is_prerelease or ver.is_postrelease:
                        continue
                    latest_version = max(latest_version, ver)
                if latest_version > version:
                    logging.warning(
                        '\nDeepCTR version {0} detected. Your version is {1}.\nUse `pip install -U deepctr` to upgrade.Changelog: https://github.com/shenweichen/DeepCTR/releases/tag/v{0}'.format(
                            latest_version, version))
        except:
            print("Please check the latest version manually on https://pypi.org/project/deepctr/#history")
            return

    Thread(target=check, args=(version,)).start()
