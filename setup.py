from codecs import open
from os.path import abspath, dirname, join

from setuptools import find_packages, setup


def version():
    with open("autofit/__init__.py") as f:
        lines = f.read().split("\n")
    for line in lines:
        if "__version__" in line:
            return line.split("=")[1].strip(" '\"")


this_dir = abspath(dirname(__file__))
with open(join(this_dir, "README.rst"), encoding="utf-8") as file:
    long_description = file.read()

with open(join(this_dir, "requirements.txt")) as f:
    requirements = f.read().split("\n")

setup(
    name="autofit",
    version=version(),
    description="Classy non-linear optimisation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rhayes777/AutoFit",
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
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="cli",
    packages=find_packages(exclude=["docs"]),
    install_requires=requirements,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
