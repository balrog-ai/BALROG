# Contributions

We deeply encourage any contributions to our library! These can include:
- New baseline agents
- New environments
- Bugfixes and quality of life improvements.

When implementing majour features:
1. Branch out from develop
2. Open PR from your feature branch to develop, describing what you are working on
3. Implement feature on feature branch
4. Once feature is ready, solve conflicts and ask for review
5. Merge when approved (replying with LGTM is fine fow now)

Before commiting any changes, please run the following:

```
pip install black isort flake8 pre-commit
pre-commit install
pre-commit run --all-files
``` 