########################################################################################
#
# DO NOT MODIFY!!!
# If necessary, override the corresponding variable and/or target, or create new ones
# in one of the following files, depending on the nature of the override :
#
# `Makefile.variables`, `Makefile.targets` or `Makefile.private`,
#
# The only valid reason to modify this file is to fix a bug or to add new
# files to include.
########################################################################################
PROJECT_PATH := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
MAKEFILE_VERSION := 0.7.1
BUMP_TOOL := bump-my-version
BUMP_CONFIG_FILE := $(PROJECT_PATH).bumpversion.toml

# Colors
_SECTION := \033[1m\033[34m
_TARGET  := \033[36m
_NORMAL  := \033[0m

.DEFAULT_GOAL := help
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
version: ## display current version of makefile
	@echo "version: $(MAKEFILE_VERSION)"

## -- Makefile versioning targets --------------------------------------------------- ##

# Use the "dry" target for a dry-run version bump, ex.
# make bump-major dry
BUMP_ARGS ?= --verbose
ifeq ($(filter dry, $(MAKECMDGOALS)), dry)
	BUMP_ARGS := $(BUMP_ARGS) --dry-run --allow-dirty
endif

.PHONY: dry
dry: ## Add the dry target for a preview of changes; ex. `make bump-major dry`
	@-echo > /dev/null

.PHONY: bump-major
bump-major: ## Bump makefile major version  <X.0.0>
	$(BUMP_TOOL) bump $(BUMP_ARGS) major --config-file $(BUMP_CONFIG_FILE)

.PHONY: bump-minor
bump-minor: ## Bump makefile minor version  <0.X.0>
	$(BUMP_TOOL) bump $(BUMP_ARGS) minor --config-file $(BUMP_CONFIG_FILE)

.PHONY: bump-patch
bump-patch: ## Bump makefile patch version  <0.0.X>
	$(BUMP_TOOL) bump $(BUMP_ARGS) patch --config-file $(BUMP_CONFIG_FILE)
