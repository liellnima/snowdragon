########################################################################################
# DO NOT MODIFY!!!
# If necessary, override the corresponding variable and/or target, or create new ones
# in one of the following files, depending on the nature of the override :
#
# Makefile.variables, Makefile.targets or Makefile.private,
#
# The only valid reason to modify this file is to fix a bug or to add new
# files to include.
#
# Please report bugs to francis.pelletier@mila.quebec
########################################################################################

# Basic variables
PROJECT_PATH := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
MAKEFILE_NAME := $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))
SHELL := /usr/bin/env bash
BUMP_TOOL := bump-my-version
MAKEFILE_VERSION := 0.7.1
DOCKER_COMPOSE ?= docker compose
AUTO_INSTALL ?=

# Conda variables
# CONDA_TOOL can be overridden in Makefile.private file
CONDA_TOOL := conda
CONDA_ENVIRONMENT ?=
CONDA_YES_OPTION ?=

# Default environment to install package
# Can be overridden in Makefile.private file
DEFAULT_INSTALL_ENV ?=
DEFAULT_POETRY_INSTALL_ENV ?=

# Colors
_SECTION := \033[1m\033[34m
_TARGET  := \033[36m
_NORMAL  := \033[0m

.DEFAULT_GOAL := help

# Project and Private variables and targets import to override variables for local
# This is to make sure, sometimes the Makefile includes don't work.
-include Makefile.variables
-include Makefile.private
## -- Informative targets ------------------------------------------------------------------------------------------- ##

.PHONY: all
all: help

# Auto documented help targets & sections from comments
#	detects lines marked by double #, then applies the corresponding target/section markup
#   target comments must be defined after their dependencies (if any)
#	section comments must have at least a double dash (-)
#
# 	Original Reference:
#		https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# 	Formats:
#		https://misc.flogisoft.com/bash/tip_colors_and_formatting
#
#	As well as influenced by it's implementation in the Weaver Project
#		https://github.com/crim-ca/weaver/tree/master

.PHONY: help
# note: use "\#\#" to escape results that would self-match in this target's search definition
help: ## print this help message (default)
	@echo ""
	@echo "Please use 'make <target>' where <target> is one of below options."
	@echo ""
	@for makefile in $(MAKEFILE_LIST); do \
        grep -E '\#\#.*$$' "$(PROJECT_PATH)/$${makefile}" | \
            awk 'BEGIN {FS = "(:|\-\-\-)+.*\#\# "}; \
            	/\--/ {printf "$(_SECTION)%s$(_NORMAL)\n", $$1;} \
				/:/  {printf "    $(_TARGET)%-24s$(_NORMAL) %s\n", $$1, $$2} ' 2>/dev/null ; \
    done

.PHONY: targets
targets: help

.PHONY: version
version: ## display current version
	@echo "version: $(APP_VERSION)"

## -- Virtualenv targets -------------------------------------------------------------------------------------------- ##

VENV_PATH := $(PROJECT_PATH).venv
VENV_ACTIVATE := $(VENV_PATH)/bin/activate

.PHONY: venv-create
venv-create: ## Create a virtualenv '.venv' at the root of the project folder 
	@virtualenv $(VENV_PATH)
	@make -s venv-activate

.PHONY: venv-activate
venv-activate: ## Print out the shell command to activate the project's virtualenv.
	@echo "source $(VENV_ACTIVATE)"

.PHONY: venv-remove
venv-remove: ## Delete the virtualenv '.venv' at the root of the project folder.
	@if [ -d $(VENV_PATH) ]; then \
  	  echo "Current venv folder is [$(VENV_PATH)]"; \
  	  if [ "$(AUTO_INSTALL)" = "true" ]; then \
			ans="y";\
	  else \
	    echo ""; \
		echo -n "Would you like to completely delete this virtual environment? [y/N]: "; \
		read ans; \
	  fi; \
	  case $$ans in \
			[Yy]*) \
				echo ""; \
				echo "Starting deletion process for [$(VENV_PATH)]"; \
				rm -rf $(VENV_PATH); \
				echo ""; \
				echo "-- Deletion complete --"; \
				;; \
			*) \
	    		echo ""; \
				echo "Skipping virtual environment deletion."; \
				echo " "; \
				;; \
		esac; \
  	else \
  	  echo "Venv [$(VENV_PATH)] does not exist, nothing to do"; \
  	fi;

## -- Versioning targets -------------------------------------------------------------------------------------------- ##

# Use the "dry" target for a dry-run version bump, ex.
# make bump-major dry
BUMP_ARGS ?= --verbose
ifeq ($(filter dry, $(MAKECMDGOALS)), dry)
	BUMP_ARGS := $(BUMP_ARGS) --dry-run --allow-dirty
endif

.PHONY: dry
dry: ## Add the dry target for a preview of changes; ex. 'make bump-major dry'
	@-echo > /dev/null

.PHONY: bump-major
bump-major: ## Bump application major version  <X.0.0>
	$(BUMP_TOOL) bump $(BUMP_ARGS) major

.PHONY: bump-minor
bump-minor: ## Bump application minor version  <0.X.0>
	$(BUMP_TOOL) bump $(BUMP_ARGS) minor

.PHONY: bump-patch
bump-patch: ## Bump application patch version  <0.0.X>
	$(BUMP_TOOL) bump $(BUMP_ARGS) patch



