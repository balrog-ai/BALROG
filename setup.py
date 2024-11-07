import setuptools
from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()


def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#") and not line.startswith("-e")]


setup(
    # Information
    name="balrog",
    description="Benchmark for In Context Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.0",
    url="https://github.com/DavidePaglieri/BALROG/",
    author="Davide Paglieri",
    license="MIT",
    keywords="reinforcement learning ai nlp llm",
    project_urls={
        "website": "https://www.balrogai.com/",
    },
    install_requires=[parse_requirements("requirements.txt")],
    entry_points={
        "console_scripts": [
            "balrog-post-install=post_install:main",
        ],
    },
    extras_require={"dev": ["black", "isort>=5.12", "pytest<8.0", "flake8", "pre-commit", "twine"]},
    package_dir={"": "./"},
    packages=setuptools.find_packages(where="./", include=["balrog*"]),
    include_package_data=True,
    python_requires=">=3.8",
)
