########################################################################################
#
# MODIFY WITH CARE!!!
# If necessary, override the corresponding variable and/or target, or create new ones
# in one of the following files, depending on the nature of the override :
#
# `Makefile.variables`, `Makefile.targets` or `Makefile.private`,
#
# The only valid reason to modify this file is to fix a bug or to add/remove
# files to include.
#
# REMEMBER!!!
# This is a project level config, any changes here will affect all other users
#
########################################################################################
#
# Necessary make files
#
include .make/base.make

#
# Optional makefiles - Comment/uncomment the targets you want for the project
#
## Conda targets
include .make/conda.make

## Poetry targets - !!! If using Poetry, you should comment out the UV file below and consider
## if you need the conda file above !!!
include .make/poetry.make

## UV targets - !!! If using UV, you should comment out the poetry file above, and possibly the conda file too !!!
#include .make/uv.make

## Linting targets
include .make/lint.make

## Test related targets
include .make/test.make

#
# Project related makefiles
#
## Custom targets and variables
-include Makefile.targets
-include Makefile.variables

## Private variables and targets import to override variables for local
-include Makefile.private
