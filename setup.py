#!/usr/bin/env python3
# pip install -r requirements.txt .
__author__ = 'Will Rowe'
__mail__ = "will.rowe@stfc.ac.uk"

from setuptools import setup
from banner._version import __version__

# run the setup
setup(
    name = "banner",
    version=__version__,
    packages = ["banner"],
    author = 'Will Rowe',
    author_email = 'will.rowe@stfc.ac.uk',
    url = 'http://will-rowe.github.io/',
    description = 'banner is a tool for predicting microbiome labels based on hulk sketches',
    long_description = open('README.md').read(),
    license = 'MIT',
    package_dir = {'banner': 'banner'},
    entry_points = {
        "console_scripts": ['banner = banner.main:Banner']
        },
    tests_require=['pytest'],
)
