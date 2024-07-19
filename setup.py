import os
from codecs import open
from os.path import abspath, dirname, join
from os import environ

from setuptools import find_packages, setup

this_dir = abspath(dirname(__file__))
with open(join(this_dir, "README.rst"), encoding="utf-8") as file:
    long_description = file.read()

with open(join(this_dir, "requirements.txt")) as f:
    requirements = f.read().split("\n")

version = environ.get("VERSION", "2024.07.16.1")
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
    name="autofit",
    version=version,
    description="Classy Probabilistic Programming",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/rhayes777/PyAutoFit",
    author="James Nightingale and Richard Hayes",
    author_email="richard@rghsoftware.co.uk",
    include_package_data=True,
    license="MIT License",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires='>=3.7',
    keywords="cli",
    packages=find_packages(exclude=["docs", "test_autofit", "test_autofit*"]) + config_packages('autofit/config'),
    install_requires=requirements,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
