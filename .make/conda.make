

## -- Conda targets ------------------------------------------------------------------------------------------------- ##

CONDA_ENVIRONMENT_FILE := environment.yml
.PHONY: conda-install
conda-install: ## Install Miniconda on your local machine
	@echo "Looking for [$(CONDA_TOOL)]..."; \
	$(CONDA_TOOL) --version; \
	if [ $$? != "0" ]; then \
		echo " "; \
		echo "Your defined Conda tool [$(CONDA_TOOL)] has not been found."; \
		echo " "; \
		echo "If you know you already have [$(CONDA_TOOL)] or some other Conda tool installed,"; \
		echo "Check your [CONDA_TOOL] variable in the Makefile.private for typos."; \
		echo " "; \
		echo "If your conda tool has not been initiated through your .bashrc file,"; \
		echo "consider using the full path to its executable instead when"; \
		echo "defining your [CONDA_TOOL] variable"; \
		echo " "; \
		echo "If in doubt, don't install Conda and manually create and activate"; \
		echo "your own Python environment."; \
		echo " "; \
		echo "It is strongly NOT advisable to execute this command if you are on a"; \
		echo "Compute Cluster (ie. Mila/DRAC), as they either have modules available (Mila),"; \
		echo "or even prohibit the installation and use of Conda (DRAC) based environments."; \
		echo " "; \
		echo -n "Would you like to install and initialize Miniconda ? [y/N]: "; \
		read ans; \
		case $$ans in \
			[Yy]*) \
				echo "Fetching and installing miniconda"; \
				echo " "; \
				wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh; \
    			bash ~/miniconda.sh -b -p $${HOME}/.conda; \
    			export PATH=$${HOME}/.conda/bin:$$PATH; \
    			conda init; \
				/usr/bin/rm ~/miniconda.sh; \
				;; \
			*) \
				echo "Skipping installation."; \
				echo " "; \
				;; \
		esac; \
	else \
		echo "Conda tool [$(CONDA_TOOL)] has been found, skipping installation"; \
	fi;

.PHONY: mamba-install
mamba-install: ## Install Micromamba on you local machine
	@echo "Looking for [micromamba]..."; \
	micromamba --version; \
	if [ $$? != "0" ]; then \
	  	echo ""; \
	  	echo "[micromamba] has not been found."; \
	  	echo ""; \
	  	echo "If you know you already have installed [micromamba] installed, it might not have"; \
	  	echo "been properly activated (which can also be on purpose)."; \
	  	echo ""; \
	  	echo "If your Conda/Micromamba tool has not been initiated through your .bashrc file,"; \
		echo "consider using the full path to its executable instead when"; \
		echo "defining your [CONDA_TOOL] variable"; \
		echo " "; \
		echo "If you do decide to install Micromamba, please take care to define the [CONDA_TOOL]"; \
		echo "variable in you personal 'Makefile.private' file as CONDA_TOOL := micromamba."; \
		echo " "; \
		echo "If in doubt, don't install Micromamba and manually create and activate"; \
		echo "your own Python environment."; \
	  	echo ""; \
	  	echo "It is strongly NOT advisable to execute this command if you are on a"; \
		echo "Compute Cluster (ie. Mila/DRAC), as they either have modules available (Mila),"; \
		echo "or even prohibit the installation and use of Conda based environments (DRAC)."; \
	  	echo ""; \
		echo -n "Would you like to install and initialize [micromamba] ? [y/N]: "; \
		read ans; \
		case $$ans in \
			[Yy]*) \
				echo 'Installing Micromamba'
				wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba;\
				mv bin/micromamba ~/.local/bin/micromamba;\
				rm -rf bin/;\
				~/.local/bin/micromamba shell init -s bash ~/.micromamba;\
				;; \
			*) \
				echo "Skipping installation."; \
				echo " "; \
				;; \
		esac; \


.PHONY: conda-create-env
conda-create-env: conda-install ## Create a local Conda environment based on 'environment.yml' file
	@$(CONDA_TOOL) env create $(CONDA_YES_OPTION) -f $(CONDA_ENVIRONMENT_FILE)

.PHONY: conda-env-info
conda-env-info: ## Print information about active Conda environment using <CONDA_TOOL>
	@$(CONDA_TOOL) info

.PHONY: conda-activate
conda-activate: ## Print the shell command to activate the project's Conda env.
	@echo "$(CONDA_TOOL) activate $(CONDA_ENVIRONMENT)"

.PHONY: _conda-poetry-install
_conda-poetry-install:
	@$(CONDA_TOOL) run -n $(CONDA_ENVIRONMENT) python --version;  \
	if [ $$? != "0" ]; then \
		echo "Target environment doesn't seem to exist..."; \
		if [ "$(AUTO_INSTALL)" = "true" ]; then \
				ans="y";\
		else \
			echo ""; \
			echo -n "Do you want to create it? [y/N] "; \
			read ans; \
		fi; \
		case $$ans in \
			[Yy]*) \
				echo "Creating conda environment : [$(CONDA_ENVIRONMENT)]"; \
				make -s conda-create-env; \
				;; \
			*) \
				echo "Exiting..."; \
				exit 1;\
				;; \
		esac;\
	fi;
	$(CONDA_TOOL) run -n $(CONDA_ENVIRONMENT) $(CONDA_TOOL) install $(CONDA_YES_OPTION) -c conda-forge poetry; \
	CURRENT_VERSION=$$($(CONDA_TOOL) run -n $(CONDA_ENVIRONMENT) poetry --version | awk '{print $$NF}' | tr -d ')'); \
	REQUIRED_VERSION="1.6.0"; \
	if [ "$$(printf '%s\n' "$$REQUIRED_VERSION" "$$CURRENT_VERSION" | sort -V | head -n1)" != "$$REQUIRED_VERSION" ]; then \
		echo "Poetry installed version $$CURRENT_VERSION is less than minimal version $$REQUIRED_VERSION, fixing urllib3 version to prevent problems"; \
		$(CONDA_TOOL) run -n $(CONDA_ENVIRONMENT)  poetry add "urllib3<2.0.0"; \
	fi;

.PHONY:conda-poetry-install
conda-poetry-install: ## Install Poetry in the project's Conda environment. Will fail if Conda is not found
	@poetry --version; \
    	if [ $$? != "0" ]; then \
			echo "Poetry not found, proceeding to install Poetry..."; \
			echo "Looking for [$(CONDA_TOOL)]...";\
			$(CONDA_TOOL) --version; \
			if [ $$? != "0" ]; then \
				echo "$(CONDA_TOOL) not found; Poetry will not be installed"; \
			else \
				echo "Installing Poetry with Conda in [$(CONDA_ENVIRONMENT)] environment"; \
				make -s _conda-poetry-install; \
			fi; \
		else \
			echo ""; \
			echo "Poetry has been found on this system :"; \
			echo "    Install location: $$(which poetry)"; \
			echo ""; \
			if [ "$(AUTO_INSTALL)" = "true" ]; then \
				ans="y";\
			else \
				echo -n "Would you like to install poetry in the project's conda environment anyway ? [y/N]: "; \
				read ans; \
			fi; \
			case $$ans in \
				[Yy]*) \
					echo "Installing Poetry with Conda in [$(CONDA_ENVIRONMENT)] environment"; \
					make -s _conda-poetry-install; \
					;; \
				*) \
					echo "Skipping installation."; \
					echo " "; \
					;; \
			esac; \
		fi;

.PHONY: conda-poetry-uninstall
conda-poetry-uninstall: ## Uninstall Poetry located in currently active Conda environment
	$(CONDA_TOOL) run -n $(CONDA_ENVIRONMENT) $(CONDA_TOOL) remove $(CONDA_YES_OPTION) poetry

.PHONY: conda-clean-env
conda-clean-env: ## Completely removes local project's Conda environment
	$(CONDA_TOOL) env remove $(CONDA_YES_OPTION) -n $(CONDA_ENVIRONMENT)
