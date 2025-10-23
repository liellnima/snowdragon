## -- Linting targets ----------------------------------------------------------------------------------------------- ##

.PHONY: check-lint
check-lint: ## Check code linting (black, isort, flake8, docformatter and pylint)
	$(ENV_COMMAND_TOOL) nox -s check

.PHONY: check-pylint
check-pylint: ## Check code with pylint
	$(ENV_COMMAND_TOOL) nox -s pylint

.PHONY: check-complexity
check-complexity: ## Check code cyclomatic complexity with Flake8-McCabe
	$(ENV_COMMAND_TOOL) nox -s complexity

.PHONY: fix-lint
fix-lint: ## Fix code linting (autoflake, autopep8, black, isort, flynt, docformatter)
	$(ENV_COMMAND_TOOL) nox -s fix

.PHONY: markdown-lint
markdown-lint: ## Fix markdown linting using mdformat
	$(ENV_COMMAND_TOOL) nox -s mdformat

.PHONY: precommit
precommit: ## Run Pre-commit on all files manually
	$(ENV_COMMAND_TOOL) nox -s precommit

.PHONY: ruff
ruff: ## Run the ruff linter
	$(ENV_COMMAND_TOOL) nox -s ruff-lint

.PHONY: ruff-fix
ruff-fix: ## Run the ruff linter and fix automatically fixable errors
	$(ENV_COMMAND_TOOL) nox -s ruff-fix

.PHONY: ruff-format
ruff-format: ## Run the ruff code formatter
	$(ENV_COMMAND_TOOL) nox -s ruff-format
