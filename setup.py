import setuptools
from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    # Information
    name="balrog",
    description="Benchmark for Agentic LLM and VLM Reasoning On Games",
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
    install_requires=[
        "openai",
        "anthropic",
        "google-generativeai",
        "hydra-core",
        "opencv-python-headless",
        "wandb",
        "pytest",
        "crafter",
        "gym==0.23",
        "requests",
        "balrog-nle",
        "minihack @ git+https://github.com/balrog-ai/minihack.git",
        "textworld @ git+https://github.com/balrog-ai/TextWorld.git",
        "tatsu==5.8.3",
        "minigrid @ git+https://github.com/BartekCupial/Minigrid.git",
        "baba @ git+https://github.com/nacloos/baba-is-ai.git",
    ],
    entry_points={
        "console_scripts": [
            "balrog-post-install=balrog.scripts.post_install:main",
        ],
    },
    extras_require={
        "dev": [
            "black",
            "isort>=5.12",
            "pytest<8.0",
            "flake8",
            "pre-commit",
            "twine",
        ]
    },
    package_dir={"": "./"},
    packages=setuptools.find_packages(where="./", include=["balrog*"]),
    package_data={
        "balrog": [
            "config/config.yaml",
            "environments/nle/achievements.json",
            "environments/nle/Hack-Regular.ttf",
            "environments/nle/tiles.pkl",
            "environments/nle/Tiles16x16.png",
        ]
    },
    include_package_data=True,
    python_requires=">=3.8",
)
