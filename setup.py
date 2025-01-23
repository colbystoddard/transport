from setuptools import setup

with open("VERSION") as file:
    version = file.readline()

requirements = [
        "numpy>=1.22.4",
        "scipy>=1.7.3"
        ]

setup(
        name="transport",
        py_modules = ["transport"],
        version = version,
        install_requires=requirements
        )
