import os
from codecs import open
from os.path import abspath, dirname, join
from os import environ

from setuptools import setup

version = environ.get("VERSION", "1.0.dev0")
this_dir = abspath(dirname(__file__))

with open(join(this_dir, "requirements.txt")) as f:
    requirements = f.read().split("\n")

requirements.extend([
    f'autoconf=={version}'
])


def config_packages(directory):
    paths = [directory.replace("/", ".")]
    for (path, directories, filenames) in os.walk(directory):
        for directory in directories:
            paths.append(f'{path}/{directory}'.replace("/", "."))
    return paths


setup(
    version=version,
    install_requires=requirements,
)
