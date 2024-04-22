from os import path
from setuptools import setup

ROOT = path.abspath(path.dirname(__file__))

# Get requirements from file
with open(path.join(ROOT, "requirements.txt")) as f:
    requirements = [line.strip() for line in f.readlines() if not line.startswith("#")]

setup(
    name="snowdragon",
    version="2.0",
    description="Automatic snow layer classification from SMP measurements.",
    author="Julia Kaltenborn",
    author_email="julia.kaltenborn@mail.mcgill.ca",
    install_requires=requirements,
    python_requires=">=3.10", 
)

# add something to rewrite import in keras_self_attention:
# from tensorflow import keras
# instead of: import keras
