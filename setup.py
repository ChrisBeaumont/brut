#!/usr/bin/env python

from setuptools import setup, Command

from distutils.command.build_py import build_py

setup(name='bubbly',
      version='0.0.1',
      description='Machine Learning on the Milky Way Project datataset',
      author='Chris Beaumont',
      author_email='cbeaumont@cfa.harvard.edu',
      packages=['bubbly'],
      cmdclass={'build_py': build_py},
      keywords=['Scientific/Engineering'],
      classifiers=[
          "Programming Language :: Python",
          "License :: OSI Approved :: MIT License",
          ],
    )
