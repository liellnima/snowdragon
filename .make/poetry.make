# Project and Private variables and targets import to override variables for local
# This is to make sure, sometimes the Makefile includes don't work.
-include Makefile.variables
-include Makefile.private

ENV_COMMAND_TOOL := poetry run

## -- Poetry targets ------------------------------------------------------------------------------------------------ ##

.PHONY: poetry-install-auto
poetry-install-auto: ## Install Poetry automatically using DEFAULT_POETRY_INSTALL_ENV. Defaults to venv install
	@poetry --version; \
    	if [ $$? != "0" ]; then \
			echo "Poetry not found, proceeding to install Poetry..."; \
			if [ "$(DEFAULT_POETRY_INSTALL_ENV)" == "conda" ]; then \
			    echo ""; \
			    echo "[DEFAULT_POETRY_INSTALL_ENV] is defined as 'conda', installing poetry with the 'poetry-install-conda' target"; \
			    echo ""; \
				ans_where="conda"; \
			elif [ "$(DEFAULT_POETRY_INSTALL_ENV)" == "venv" ]; then \
				echo ""; \
				echo "[DEFAULT_POETRY_INSTALL_ENV] is defined as 'venv', installing poetry with the 'poetry-install-venv' target"; \
				echo ""; \
				ans_where="venv"; \
			else\
				echo ""; \
				echo "[DEFAULT_POETRY_INSTALL_ENV] is not defined, defaulting to installing poetry with the 'poetry-install-venv' target"; \
				echo ""; \
				ans_where="venv"; \
			fi; \
			case $$ans_where in \
				"venv" | "Venv" |"VENV") \
					make AUTO_INSTALL=true -s poetry-install-venv; \
					;; \
				"conda" | "Conda" | "CONDA") \
					echo "Installing poetry with Conda"; \
					make AUTO_INSTALL=true -s conda-poetry-install; \
					;; \
				*) \
					echo ""; \
					echo -e "\e[1;39;41m-- WARNING --\e[0m There was an unexpected error. Option $$ans_how not found, exiting process."; \
					echo ""; \
					exit 1; \
			esac; \
		fi;

.PHONY: _pipx_install_poetry
_pipx_install_poetry:
	@output="$$(pip install poetry --dry-run)"; \
	if echo "$$output" | grep -q computecanada ; then \
		echo ""; \
		echo -e "\e[1;39;41m-- WARNING --\e[0m Compute Canada (DRAC) environment detected: Installing Poetry < 2.0.0"; \
		echo ""; \
		pipx install 'poetry<2.0.0' ; \
	else \
		pipx install poetry ; \
	fi;


.PHONY: poetry-install
poetry-install: ## Install Poetry interactively.
	@echo "Looking for Poetry version...";\
	poetry --version; \
	if [ $$? != "0" ]; then \
		if [ "$(AUTO_INSTALL)" = "true" ]; then \
			ans="y";\
		else \
			echo "Poetry not found..."; \
			echo "Looking for pipx version...";\
			pipx_found=0; \
			pipx --version; \
				if [ $$? != "0" ]; then \
					pipx_found=1; \
					echo "pipx not found..."; \
					echo""; \
					echo -n "Would you like to install pipx and Poetry? [y/N]: "; \
				else \
					echo""; \
					echo -n "Would you like to install Poetry using pipx? [y/N]: "; \
				fi; \
			read ans; \
		fi; \
		case $$ans in \
			[Yy]*) \
				if [ $$pipx_found == "1" ]; then \
					echo""; \
					echo -e "\e[1;39;41m-- WARNING --\e[0m The following pip has been found and will be used to install pipx: "; \
					echo "    -> "$$(which pip); \
					echo""; \
					echo "If you do not have write permission to that environment, using it to install pipx will fail."; \
					echo "If this is the case, you should install pipx using a virtual one."; \
					echo""; \
					echo "See documentation for more information."; \
					echo""; \
					echo -n "Would you like to use the local available pip above, or create virtual environment to install pipx? [local/virtual]: "; \
					read ans_how; \
					case $$ans_how in \
						"LOCAL" | "Local" |"local") \
							make -s poetry-install-local; \
							;; \
						"VIRTUAL" | "Virtual" | "virtual") \
							make -s poetry-install-venv; \
							;; \
						*) \
							echo ""; \
							echo -e "\e[1;39;41m-- WARNING --\e[0m Option $$ans_how not found, exiting process."; \
							echo ""; \
							exit 1; \
					esac; \
				else \
					echo "Installing Poetry"; \
					make -s _pipx_install_poetry; \
				fi; \
				;; \
			*) \
				echo "Skipping installation."; \
				echo " "; \
				;; \
		esac; \
	fi;

PIPX_VENV_PATH := $$HOME/.pipx_venv
.PHONY: poetry-install-venv
poetry-install-venv: ## Install standalone Poetry. Will install pipx in $HOME/.pipx_venv
	@pipx --version; \
	if [ $$? != "0" ]; then \
		echo "Creating virtual environment using venv here : [$(PIPX_VENV_PATH)]"; \
		python3 -m venv $(PIPX_VENV_PATH); \
		echo "Activating virtual environment [$(PIPX_VENV_PATH)]"; \
		source $(PIPX_VENV_PATH)/bin/activate; \
		pip3 install pipx; \
		pipx ensurepath; \
		source $(PIPX_VENV_PATH)/bin/activate && make -s _pipx_install_poetry ; \
	else \
		make -s _pipx_install_poetry ; \
	fi;

.PHONY: poetry-install-local
poetry-install-local: ## Install standalone Poetry. Will install pipx with locally available pip.
	@pipx --version; \
	if [ $$? != "0" ]; then \
		echo "pipx not found; installing pipx"; \
		pip3 install pipx; \
		pipx ensurepath; \
	fi;
	@echo "Installing Poetry"
	@make -s _pipx_install_poetry


.PHONY: poetry-env-info
poetry-env-info: ## Information about the currently active environment used by Poetry
	@poetry env info

.PHONY: poetry-create-env
poetry-create-env: ## Create a Poetry managed environment for the project (Outside of Conda environment).
	@echo "Creating Poetry environment that will use Python $(PYTHON_VERSION)"; \
	poetry env use $(PYTHON_VERSION); \
	poetry env info
	@echo""
	@echo "This environment can be accessed either by using the <poetry run YOUR COMMAND>"
	@echo "command, or activated with the <poetry shell> command."
	@echo""
	@echo "Use <poetry --help> and <poetry list> for more information"
	@echo""

.PHONY: poetry-activate
poetry-activate: ## Print the shell command to activate the project's poetry env.
	poetry env activate

.PHONY: poetry-remove-env
poetry-remove-env: ## Remove current project's Poetry managed environment.
	@if [ "$(AUTO_INSTALL)" = "true" ]; then \
		ans_env="y";\
		env_path=$$(poetry env info -p); \
		env_name=$$(basename $$env_path); \
	else \
		echo""; \
		echo "Looking for poetry environments..."; \
		env_path=$$(poetry env info -p); \
		if [[ "$$env_path" != "" ]]; then \
			echo "The following environment has been found for this project: "; \
			env_name=$$(basename $$env_path); \
			echo""; \
			echo "Env name : $$env_name"; \
			echo "PATH     : $$env_path"; \
			echo""; \
			echo "If the active environment listed above is a Conda environment,"; \
			echo "Choosing to delete it will have no effect; use the target <make conda-clean-env>"; \
			echo""; \
			echo""; \
			echo "If the active environment listed above is a venv environment,"; \
			echo "Choosing to delete it will have no effect; use the bash command $ rm -rf <PATH_TO_VENV>"; \
			echo""; \
			echo -n "Would you like delete the environment listed above? [y/N]: "; \
			read ans_env; \
		else \
			env_name="None"; \
			env_path="None"; \
  		fi; \
	fi; \
	if [[ $$env_name != "None" ]]; then \
		case $$ans_env in \
			[Yy]*) \
				poetry env remove $$env_name || echo "No environment was removed"; \
				;; \
			*) \
				echo "No environment was found/provided - skipping environment deletion"; \
				;;\
		esac; \
	else \
		echo "No environments were found... skipping environment deletion"; \
	fi; \

.PHONY: poetry-uninstall
poetry-uninstall: poetry-remove-env ## Uninstall pipx-installed Poetry and the created environment
	@if [ "$(AUTO_INSTALL)" = "true" ]; then \
		ans="y";\
	else \
		echo""; \
		echo -n "Would you like to uninstall pipx-installed Poetry? [y/N]: "; \
		read ans; \
	fi; \
	case $$ans in \
		[Yy]*) \
			pipx --version ; \
			if [ $$? != "0" ]; then \
				echo "" ; \
				echo "Pipx not found globally, trying with $(PIPX_VENV_PATH) env" ;\
				echo "" ; \
				source $(PIPX_VENV_PATH)/bin/activate && pipx uninstall poetry ; \
			else \
				pipx uninstall poetry ; \
				fi; \
			;; \
		*) \
			echo "Skipping uninstallation."; \
			echo " "; \
			;; \
	esac; \

.PHONY: poetry-uninstall-pipx
poetry-uninstall-pipx: poetry-remove-env ## Uninstall pipx-installed Poetry, the created Poetry environment and pipx
	@if [ "$(AUTO_INSTALL)" = "true" ]; then \
		ans="y";\
	else \
		echo""; \
		echo -n "Would you like to uninstall pipx-installed Poetry and pipx? [y/N]: "; \
		read ans; \
	fi; \
	case $$ans in \
		[Yy]*) \
			pipx --version ; \
			if [ $$? != "0" ]; then \
				echo "" ; \
				echo "Pipx not found globally, trying with $(PIPX_VENV_PATH) env" ;\
				echo "" ; \
				source $(PIPX_VENV_PATH)/bin/activate && pipx uninstall poetry && pip uninstall -y pipx; \
			else \
				pipx uninstall poetry ; \
				pip uninstall -y pipx ;\
				fi; \
			;; \
		*) \
			echo "Skipping uninstallation."; \
			echo " "; \
			;; \
	esac; \

.PHONY: poetry-uninstall-venv
poetry-uninstall-venv: poetry-remove-env ## Uninstall pipx-installed Poetry, the created Poetry environment, pipx and $HOME/.pipx_venv
	@if [ "$(AUTO_INSTALL)" = "true" ]; then \
		ans="y";\
	else \
		echo""; \
		echo -n "Would you like to uninstall pipx-installed Poetry and pipx? [y/N]: "; \
		read ans; \
	fi; \
	case $$ans in \
		[Yy]*) \
			(source $(PIPX_VENV_PATH)/bin/activate && pipx uninstall poetry); \
			(source $(PIPX_VENV_PATH)/bin/activate && pip uninstall -y pipx); \
			;; \
		*) \
			echo "Skipping uninstallation."; \
			echo " "; \
			;; \
	esac; \

	@if [ "$(AUTO_INSTALL)" = "true" ]; then \
		ans="y";\
	else \
		echo""; \
		echo -n "Would you like to remove the virtual environment located here : [$(PIPX_VENV_PATH)] ? [y/N]: "; \
		read ans; \
	fi; \
	case $$ans in \
		[Yy]*) \
			rm -r $(PIPX_VENV_PATH); \
			;; \
		*) \
			echo "Skipping [$(PIPX_VENV_PATH)] virtual environment removal."; \
			echo ""; \
			;; \
	esac; \

## -- Install targets (All install targets will install Poetry if not found using 'make poetry-install-auto')-------- ##


POETRY_COMMAND := poetry

ifeq ($(DEFAULT_INSTALL_ENV),venv)
POETRY_COMMAND := source $(VENV_ACTIVATE) && poetry
else ifeq ($(DEFAULT_INSTALL_ENV),poetry)
POETRY_COMMAND := poetry
else ifeq ($(DEFAULT_INSTALL_ENV),conda)
POETRY_COMMAND := $(CONDA_TOOL) run -n $(CONDA_ENVIRONMENT) poetry
endif

.PHONY: _check-env
_check-env:
	@if ! [ $(DEFAULT_INSTALL_ENV) ]; then \
		echo -e "\e[1;39;41m-- WARNING --\e[0m No installation environment have been defined." ; \
		echo "" ; \
		echo "[DEFAULT_INSTALL_ENV] is not defined - Poetry will use the currently activated environment." ; \
		echo "If there is no currently active environment (ie. conda or venv)," ; \
		echo "Poetry will create and manage it's own environment." ; \
	elif [ $(DEFAULT_INSTALL_ENV) = "venv" ]; then \
		if [ ! -f $(VENV_ACTIVATE) ]; then \
			make -s venv-create ;\
		fi; \
	elif [ $(DEFAULT_INSTALL_ENV) = "conda" ]; then \
		if ! $(CONDA_TOOL) env list | grep -q $(CONDA_ENVIRONMENT) ; then \
			make -s conda-create-env ; \
		fi; \
	fi;

.PHONY: _remind-env-activate
_remind-env-activate:
	@echo ""
	@echo "Activate your environment using the following command:"
	@echo ""
	@if ! [ $(DEFAULT_INSTALL_ENV) ] || [ $(DEFAULT_INSTALL_ENV) = "poetry" ]; then \
		make -s poetry-activate ; \
		echo "" ; \
		echo "You can also use the eval bash command : eval \$$(make poetry-activate)"; \
		echo "" ; \
		echo "The environment can also be used through the 'poetry run <command>' command."; \
		echo "" ; \
		echo "    Ex: poetry run python <path_to_script>"; \
	elif [ $(DEFAULT_INSTALL_ENV) = "venv" ]; then \
		make -s venv-activate ; \
		echo "" ; \
		echo "You can also use the eval bash command : eval \$$(make venv-activate)"; \
	elif [ $(DEFAULT_INSTALL_ENV) = "conda" ]; then \
		make -s conda-activate ; \
		echo "" ; \
		echo "You can also use the eval bash command : eval \$$(make conda-activate)"; \
	fi;
	@echo ""

.PHONY: install
install: install-precommit ## Install the application package, developer dependencies and pre-commit hook

.PHONY: install-precommit
install-precommit: install-dev ## Install the pre-commit hooks (also installs developer dependencies)
	@if [ -f .git/hooks/pre-commit ]; then \
		echo "Pre-commit hook found"; \
	else \
	  	echo "Pre-commit hook not found, proceeding to configure it"; \
		$(POETRY_COMMAND) run pre-commit install; \
	fi;

.PHONY: install-dev
install-dev: poetry-install-auto _check-env ## Install the application along with developer dependencies
	@$(POETRY_COMMAND) install --with dev
	@make -s _remind-env-activate

.PHONY: install-with-lab
install-with-lab: poetry-install-auto _check-env ## Install the application and it's dev dependencies, including Jupyter Lab
	@$(POETRY_COMMAND) install --with dev --with lab
	@make -s _remind-env-activate


.PHONY: install-package
install-package: poetry-install-auto _check-env ## Install the application package only
	@$(POETRY_COMMAND) install
	@make -s _remind-env-activate
