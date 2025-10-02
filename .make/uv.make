# Project and Private variables and targets import to override variables for local
# This is to make sure, sometimes the Makefile includes don't work.

## -- UV targets ------------------------------------------------------------------------------------------------ ##
ENV_COMMAND_TOOL := uv run

.PHONY: uv-install-auto
uv-install-auto:
	uv --version; \
	if [ $$? != "0" ]; then \
		@make -s uv-install-venv; \
	fi;

.PHONY: uv-install
uv-install: ## Install uv interactively.
	@echo "Looking for uv version...";\
	uv --version; \
	if [ $$? != "0" ]; then \
		if [ "$(AUTO_INSTALL)" = "true" ]; then \
			ans="y";\
		else \
			echo "uv not found..."; \
			echo "Looking for pipx version...";\
			pipx_found=0; \
			pipx --version; \
				if [ $$? != "0" ]; then \
					pipx_found=1; \
					echo "pipx not found..."; \
					echo""; \
					echo -n "Would you like to install pipx and uv? [y/N]: "; \
				else \
					echo""; \
					echo -n "Would you like to install uv using pipx? [y/N]: "; \
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
							make -s uv-install-local; \
							;; \
						"VIRTUAL" | "Virtual" | "virtual") \
							make -s uv-install-venv; \
							;; \
						*) \
							echo ""; \
							echo -e "\e[1;39;41m-- WARNING --\e[0m Option $$ans_how not found, exiting process."; \
							echo ""; \
							exit 1; \
					esac; \
				else \
					echo "Installing uv"; \
					make -s _pipx_install_uv; \
				fi; \
				;; \
			*) \
				echo "Skipping installation."; \
				echo " "; \
				;; \
		esac; \
	fi;

.PHONY: _pipx_install_uv
_pipx_install_uv:
	@pipx install uv


PIPX_VENV_PATH := $$HOME/.pipx_venv
.PHONY: uv-install-venv
uv-install-venv: ## Install standalone uv. Will install pipx in $HOME/.pipx_venv
	@pipx --version; \
	if [ $$? != "0" ]; then \
		echo "Creating virtual environment using venv here : [$(PIPX_VENV_PATH)]"; \
		python3 -m venv $(PIPX_VENV_PATH); \
		echo "Activating virtual environment [$(PIPX_VENV_PATH)]"; \
		source $(PIPX_VENV_PATH)/bin/activate; \
		pip3 install pipx; \
		pipx ensurepath; \
		source $(PIPX_VENV_PATH)/bin/activate && make -s _pipx_install_uv ; \
	else \
		make -s _pipx_install_uv ; \
	fi;

.PHONY: uv-install-local
uv-install-local: ## Install standalone uv. Will install pipx with locally available pip.
	@pipx --version; \
	if [ $$? != "0" ]; then \
		echo "pipx not found; installing pipx"; \
		pip3 install pipx; \
		pipx ensurepath; \
	fi;
	@echo "Installing UV"
	@make -s _pipx_install_uv

.PHONY: uv-create-env
uv-create-env: ## Create a virtual environment for uv, using the project's python version.
	@uv python install $(PYTHON_VERSION)
	@uv venv --python $(PYTHON_VERSION)

.PHONY: uv-activate
uv-activate: ## Print out the shell command to activate the project's uv environment.
	@make -s venv-activate

.PHONY: uv-remove-env
uv-remove-env: ## Remove current project's uv managed environment.
	@make -s venv-remove

.PHONY: uv-uninstall
uv-uninstall: uv-remove-env ## Uninstall pipx-installed uv and the created environment
	@if [ "$(AUTO_INSTALL)" = "true" ]; then \
		ans="y";\
	else \
		echo""; \
		echo -n "Would you like to uninstall pipx-installed uv? [y/N]: "; \
		read ans; \
	fi; \
	case $$ans in \
		[Yy]*) \
			pipx --version ; \
			if [ $$? != "0" ]; then \
				echo "" ; \
				echo "Pipx not found globally, trying with $(PIPX_VENV_PATH) env" ;\
				echo "" ; \
				source $(PIPX_VENV_PATH)/bin/activate && pipx uninstall uv ; \
			else \
				pipx uninstall uv ; \
				fi; \
			;; \
		*) \
			echo "Skipping uninstallation."; \
			echo " "; \
			;; \
	esac; \

.PHONY: uv-uninstall-pipx
uv-uninstall-pipx: uv-remove-env ## Uninstall pipx-installed uv, the created uv environment and pipx
	@if [ "$(AUTO_INSTALL)" = "true" ]; then \
		ans="y";\
	else \
		echo""; \
		echo -n "Would you like to uninstall pipx-installed uv and pipx? [y/N]: "; \
		read ans; \
	fi; \
	case $$ans in \
		[Yy]*) \
			pipx --version ; \
			if [ $$? != "0" ]; then \
				echo "" ; \
				echo "Pipx not found globally, trying with $(PIPX_VENV_PATH) env" ;\
				echo "" ; \
				source $(PIPX_VENV_PATH)/bin/activate && pipx uninstall uv && pip uninstall -y pipx; \
			else \
				pipx uninstall uv ; \
				pip uninstall -y pipx ;\
				fi; \
			;; \
		*) \
			echo "Skipping uninstallation."; \
			echo " "; \
			;; \
	esac; \

.PHONY: uv-uninstall-venv
uv-uninstall-venv: uv-remove-env ## Uninstall pipx-installed uv, the created uv environment, pipx and $HOME/.pipx_venv
	@if [ "$(AUTO_INSTALL)" = "true" ]; then \
		ans="y";\
	else \
		echo""; \
		echo -n "Would you like to uninstall pipx-installed uv and pipx? [y/N]: "; \
		read ans; \
	fi; \
	case $$ans in \
		[Yy]*) \
			(source $(PIPX_VENV_PATH)/bin/activate && pipx uninstall uv); \
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

PHONY: uv-migrate-from-poetry
uv-migrate-from-poetry: ## Migrate the project's default poetry 'pyproject.toml' to uv.
	@if [ "$(AUTO_INSTALL)" = "true" ]; then \
		ans="y";\
	else \
		echo""; \
		echo -n "Would you like to convert your current pyproject.toml file to use uv instead of poetry ? [y/N]: "; \
		read ans; \
	fi; \
	case $$ans in \
		[Yy]*) \
		  	echo "Creating backup copy of current pyproject.toml file"; \
		  	cp pyproject.toml pyproject.toml.poetry.backup ; \
		  	echo "Migrating pyproject.toml file to use uv"; \
			uvx migrate-to-uv --dependency-groups-strategy keep-existing; \
			if [ -e pyproject.toml.uv.backup ]; then \
				echo "pyproject.toml.uv.backup file found. Proceeding to delete it."; \
			  	rm pyproject.toml.uv.backup ; \
			fi; \
			;; \
		*) \
			echo "Skipping pyproject.toml migration."; \
			echo ""; \
			;; \
	esac; \

PHONY: uv-migrate-undo
uv-migrate-undo: ## Undo previous migration of the project's default poetry 'pyproject.toml' to uv.
	@if [ "$(AUTO_INSTALL)" = "true" ]; then \
		ans="y";\
	else \
		echo""; \
		echo -n "Would you like to revert your current pyproject.toml file to use the previous backup of the file ? [y/N]: "; \
		read ans; \
	fi; \
	case $$ans in \
		[Yy]*) \
		  	echo "Checking if backup pyproject.toml file exists"; \
		  	if [ -e "$(PROJECT_PATH)/pyproject.toml.poetry.backup" ]; then \
		  	  	echo "Backup file found"; \
		  	  	echo "Making backup of current uv pyproject.toml"; \
				cp "$(PROJECT_PATH)/pyproject.toml" "$(PROJECT_PATH)/pyproject.toml.uv.backup"  ; \
		  	  	echo "Reverting pyproject.toml file to previous poetry backup."; \
				cp "$(PROJECT_PATH)pyproject.toml.poetry.backup" "$(PROJECT_PATH)/pyproject.toml"  ; \
		  	  	echo "Removing previous poetry backup."; \
				rm "$(PROJECT_PATH)/pyproject.toml.poetry.backup" ; \
		  	  	echo "Removing uv.lock file."; \
				rm "$(PROJECT_PATH)/uv.lock" ; \
			else \
				echo ""; \
				echo "Backup file not found. Skipping migration undo"; \
			fi; \
			;; \
		*) \
			echo "Skipping pyproject.toml migration."; \
			echo ""; \
			;; \
	esac; \


## -- Install targets (All install targets will install uv if not found using 'make uv-install-auto')---------------- ##
.PHONY: _remind-env-activate
_remind-env-activate:
	@echo ""
	@echo "Activate your environment using the following command:"
	@echo ""
	@make -s uv-activate
	@echo ""
	@echo "You can also use the following command line to interact with the environment"
	@echo ""
	@echo "  $$ uv run <command>"
	@echo ""

.PHONY: install
install: install-precommit ## Install the application package, developer dependencies and pre-commit hook

.PHONY: install-precommit
install-precommit: install-dev ## Install the pre-commit hooks (also installs developer dependencies)
	@if [ -f .git/hooks/pre-commit ]; then \
		echo "Pre-commit hook found"; \
	else \
	  	echo "Pre-commit hook not found, proceeding to configure it"; \
		$(ENV_COMMAND_TOOL) pre-commit install; \
	fi;

.PHONY: install-dev
install-dev: uv-install-auto ## Install the application along with developer dependencies
	@uv sync --group dev
	@make -s _remind-env-activate

.PHONY: install-with-lab
install-with-lab: uv-install-auto ## Install the application and it's dev dependencies, including Jupyter Lab
	@uv --group dev --group lab
	@make -s _remind-env-activate


.PHONY: install-package
install-package: uv-install-auto ## Install the application package only
	@uv sync
	@make -s _remind-env-activate
