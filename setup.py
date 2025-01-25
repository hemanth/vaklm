from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='vaklm',
    version='0.2.4',
    packages=find_packages(),
    install_requires=['requests'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hemamth/vaklm",  # Replace with your actual repo URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)