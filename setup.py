import setuptools
from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()
    descr_lines = long_description.split("\n")
    descr_no_gifs = []  # gifs are not supported on PyPI web page
    for dl in descr_lines:
        if not ("<img src=" in dl and "gif" in dl):
            descr_no_gifs.append(dl)

    long_description = "\n".join(descr_no_gifs)


_docs_deps = [
    "mkdocs-material",
    "mkdocs-minify-plugin",
    "mkdocs-redirects",
    "mkdocs-git-revision-date-localized-plugin",
    "mkdocs-git-committers-plugin-2",
    "mkdocs-git-authors-plugin",
]

setup(
    # Information
    name="BALROG",
    description="Benchmark for In Context Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="2.1.2",
    url="https://github.com/DavidePaglieri/BALROG/",
    author="Davide Paglieri",
    license="MIT",
    keywords="reinforcement learning ai nlp llm",
    project_urls={},
    install_requires=[
        "openai",
        "anthropic",
        "google-generativeai",
        "replicate",
        "wandb",
        "hydra-core",
        "textworld",
        "craftax",
        "gym==0.23",
    ],
    extras_require={"dev": ["black", "isort>=5.12", "pytest<8.0", "flake8", "pre-commit", "twine"] + _docs_deps},
    package_dir={"": "./"},
    packages=setuptools.find_packages(where="./", include=["balrog*"]),
    include_package_data=True,
    python_requires=">=3.8",
)
