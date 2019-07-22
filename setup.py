#!/usr/bin/env python

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import versioneer
from pathlib import Path
import sys
import os

cwd = Path(os.path.dirname(os.path.abspath(__file__)))


def open_reqs_file(file, reqs_path=Path(cwd)):
    i = 0
    while i < len(reqs):
        if reqs[i].startswith("-r"):
            reqs[i : i + 1] = open_reqs_file(reqs[i][2:].strip(), reqs_path=reqs_path)
        else:
            i += 1

    return reqs


extras_require = {}
reqs = []


def parse_requires():
    reqs_path = cwd / "requirements"
    reqs.extend(open_reqs_file("requirements.txt"))

    if sys.version_info < (3, 7):
        reqs.append("contextvars")

    for f in reqs_path.iterdir():
        extras_require[f.stem] = open_reqs_file(f.parts[-1], reqs_path=reqs_path)


parse_requires()

with open("README.md") as f:
    long_desc = f.read()


cmdclass = {"build_ext": build_cpp11_ext}
cmdclass.update(versioneer.get_cmdclass())

setup(
    name="unumpy",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    description="Array interface object for Python with pluggable backends and a multiple-dispatch"
    "mechanism for defining down-stream functions",
    url="https://github.com/Quansight-Labs/uarray/",
    maintainer="Hameer Abbasi",
    maintainer_email="habbasi@quansight.com",
    license="BSD 3-Clause License (Revised)",
    keywords="uarray,numpy,scipy,pytorch,cupy,tensorflow",
    packages=find_packages(include=["unumpy", "unumpy.*"]),
    long_description=long_desc,
    long_description_content_type="text/markdown",
    install_requires=[
        "uarray @ git+ssh://git@github.com/Quansight-Labs/uarray@master#egg=uarray"
    ],
    extras_require=extras_require,
    zip_safe=False,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    project_urls={
        "Documentation": "https://unumpy.readthedocs.io/",
        "Source": "https://github.com/Quansight-Labs/uarray/",
        "Tracker": "https://github.com/Quansight-Labs/uarray/issues",
    },
    python_requires=">=3.5, <4",
)
