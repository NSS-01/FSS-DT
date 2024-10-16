from setuptools import setup, find_packages


setup(
    name="NssMPClib",
    version="1.0",
    author="XDU_NSS",
    author_email="",
    description="",
    url="",
    license="MIT",
    packages=find_packages(where='.', exclude=(), include=('*',)),
    package_data={'': ['*.json']}
)
