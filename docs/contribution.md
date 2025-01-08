# Contribute to BALROG

## How to contribute to BALROG?

BALROG is an open source project, so all contributions and suggestions are welcome.

You can contribute in many different ways: giving ideas, answering questions, reporting bugs, proposing enhancements, 
improving the documentation, fixing bugs.

Huge thanks in advance to every contributor!


## How to work on an open Issue?

You have the list of open Issues at: https://github.com/balrog-ai/BALROG/issues

Some of them may have the label `help wanted`: that means that any contributor is welcomed!

If you would like to work on any of the open Issues:

1. Make sure it is not already assigned to someone else. You have the assignee (if any) on the top of the right column of the Issue page.

2. You can self-assign it by commenting on the Issue page with one of the keywords: `#take` or `#self-assign`.

3. Work on your self-assigned issue and eventually create a Pull Request.

## How to create a Pull Request?
1. Fork the [repository](https://github.com/balrog-ai/BALROG) by clicking on the 'Fork' button on the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

	```bash
	git clone git@github.com:<your Github handle>/BALROG.git
	cd BALROG
	git remote add upstream https://github.com/balrog-ai/BALROG.git
	```

3. Create a new branch to hold your development changes:

	```bash
	git checkout -b a-descriptive-name-for-my-changes
	```

	**do not** work on the `main` branch.

4. Set up a development environment by running the following command in a virtual (or conda) environment:

	```bash
	pip install -e '.[dev]'
	```

5. Develop the features on your branch.

6. Format your code. Run black and isort so that your newly added files look nice with the following command:

	```bash
	pre-commit install
	```

    ```
    pre-commit run --all-files
    ```

   (Optional) to update pre-commit hooks, run the following command:
	```bash
	pre-commit autoupdate
	```

7. Once you're happy with your files, add your changes and make a commit to record your changes locally:

	```bash
	git commit -m <your_commit_message>
	```

	It is a good idea to sync your copy of the code with the original
	repository regularly. This way you can quickly account for changes:

	```bash
	git fetch upstream
	git rebase upstream/main
    ```

   Push the changes to your account using:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

8. Once you are satisfied, go the webpage of your fork on GitHub. Click on "Pull request" to send your to the project maintainers for review.
