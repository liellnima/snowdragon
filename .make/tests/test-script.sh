#!/usr/bin/env bash

SCRIPT_PATH="$(readlink -f "$0")"

# Get the directory of the script
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
MAKE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$MAKE_DIR")"

TEST_ENV="$PROJECT_DIR/.testvenv"
PIPX_TEST_ENV="$PROJECT_DIR/.testvenvpipx"

# Test base.make

base() {
  echo ""
  echo "###"
  echo "### Test base targets"
  echo "###"
  echo ""
  echo "### Running 'make targets': ###"
  echo ""
  cd "$PROJECT_DIR" && make targets

  echo "### Running 'make version': ###"
  echo ""
  cd "$PROJECT_DIR" && make version

  # Test venv

  echo ""
  echo "### Running 'make venv-create' and 'make venv-remove: ###"
  echo ""
  (
    cd "$PROJECT_DIR" && \
    make VENV_PATH="$PROJECT_PATH.testvenv" venv-create && \
    make VENV_PATH="$PROJECT_PATH.testvenv" AUTO_INSTALL=true venv-remove
  )
  # Versioning

  echo ""
  echo "### Running 'make bump' targets ###"
  echo ""
  (
    cd "$PROJECT_DIR" && \
    make bump-major dry && \
    make bump-minor dry && \
    make bump-patch dry
  )
}

TEST_CONDA_ENV="lab-advanced-template-testing"
MAKEFILE_ARGS_CONDA_OVERRIDE="-f $PROJECT_DIR/Makefile -f $PROJECT_DIR/.make/conda.make"
conda(){
  echo ""
  echo "###"
  echo "### Test conda targets"
  echo "###"
  echo ""
  (
    cd "$PROJECT_DIR" &&
    make $MAKEFILE_ARGS_CONDA_OVERRIDE CONDA_ENVIRONMENT_FILE="$SCRIPT_DIR/test_environment.yml" CONDA_YES_OPTION="-y" conda-create-env && \
    make $MAKEFILE_ARGS_CONDA_OVERRIDE conda-env-info && \
    make $MAKEFILE_ARGS_CONDA_OVERRIDE CONDA_ENVIRONMENT="$TEST_CONDA_ENV" conda-activate && \
    make $MAKEFILE_ARGS_CONDA_OVERRIDE CONDA_ENVIRONMENT="$TEST_CONDA_ENV" AUTO_INSTALL=true conda-poetry-install && \
    make $MAKEFILE_ARGS_CONDA_OVERRIDE CONDA_ENVIRONMENT="$TEST_CONDA_ENV" AUTO_INSTALL=true conda-poetry-uninstall && \
    make $MAKEFILE_ARGS_CONDA_OVERRIDE CONDA_ENVIRONMENT="$TEST_CONDA_ENV" AUTO_INSTALL=true conda-clean-env
  )
}

MAKEFILE_ARGS_POETRY_OVERRIDE="-f $PROJECT_DIR/Makefile -f $PROJECT_DIR/.make/conda.make -f $PROJECT_DIR/.make/poetry.make"
lint() {
  echo ""
  echo "###"
  echo "### Test pipx poetry targets"
  echo "###"
  echo ""
  (
    cd "$PROJECT_DIR"  && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=poetry poetry-create-env && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=poetry install && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=poetry check-lint && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=poetry check-pylint && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=poetry check-complexity && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=poetry fix-lint && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=poetry precommit && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=poetry ruff && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=poetry ruff-fix && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=poetry ruff-format
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=poetry AUTO_INSTALL=true poetry-remove-env
  )
}

poetry(){
  echo ""
  echo "###"
  echo "### Test Poetry managed and venv managed installs"
  echo "###"
  echo ""
  (
    cd "$PROJECT_DIR"  && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=poetry poetry-create-env && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=poetry install && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=poetry AUTO_INSTALL=true poetry-remove-env && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=venv VENV_PATH="$TEST_ENV" install && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=venv VENV_PATH="$TEST_ENV" AUTO_INSTALL=true venv-remove
  )
  echo ""
  echo "### Test conda managed installs"
  echo ""
  (
    cd "$PROJECT_DIR" && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE CONDA_ENVIRONMENT_FILE="$SCRIPT_DIR/test_environment.yml" CONDA_YES_OPTION="-y" conda-create-env && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE DEFAULT_INSTALL_ENV=conda CONDA_ENVIRONMENT="$TEST_CONDA_ENV" install && \
    make $MAKEFILE_ARGS_POETRY_OVERRIDE CONDA_ENVIRONMENT="$TEST_CONDA_ENV" AUTO_INSTALL=true conda-clean-env
  )

}

poetry-pipx(){
  echo ""
  echo "###"
  echo "### Test pipx poetry targets"
  echo "###"
  echo ""
  (
    cd "$PROJECT_DIR" && \
    make PIPX_VENV_PATH="$PIPX_TEST_ENV" poetry-install-venv && \
    make PIPX_VENV_PATH="$PIPX_TEST_ENV" AUTO_INSTALL=true poetry-uninstall-venv && \
    make poetry-install-venv
  )
}

MAKEFILE_ARGS_UV_OVERRIDE="-f $PROJECT_DIR/Makefile -f $PROJECT_DIR/.make/uv.make"

uv(){
  echo ""
  echo "###"
  echo "### Test uv managed managed installs"
  echo "###"
  echo ""
  (
    cd "$PROJECT_DIR" && \
    make $MAKEFILE_ARGS_UV_OVERRIDE AUTO_INSTALL=true uv-migrate-from-poetry && \
    make $MAKEFILE_ARGS_UV_OVERRIDE uv-create-env && \
    make $MAKEFILE_ARGS_UV_OVERRIDE AUTO_INSTALL=true install && \
    make $MAKEFILE_ARGS_UV_OVERRIDE AUTO_INSTALL=true uv-remove-env && \
    make $MAKEFILE_ARGS_UV_OVERRIDE AUTO_INSTALL=true uv-migrate-undo && \
    rm -rf pyproject.toml.uv.backup
  )
}

uv-pipx(){
  echo ""
  echo "###"
  echo "### Test pipx uv pipx targets"
  echo "###"
  echo ""
  (
    cd "$PROJECT_DIR" && \
    make $MAKEFILE_ARGS_UV_OVERRIDE PIPX_VENV_PATH="$PIPX_TEST_ENV" uv-install-venv && \
    make $MAKEFILE_ARGS_UV_OVERRIDE PIPX_VENV_PATH="$PIPX_TEST_ENV" AUTO_INSTALL=true uv-uninstall-venv && \
    make $MAKEFILE_ARGS_UV_OVERRIDE uv-install-venv
  )
}

all() {
  base
  conda
  lint
  poetry
}

list () {
    echo
    echo "!!! Do not run this script outside of the 'lab-advanced-template' repository."
    echo "This script exists only to test the makefiles for integrity when adding or"
    echo "modifying targets !!!"
    echo
    echo " List of available tests:"
    echo
    echo "    - base   : Test 'base' targets"
    echo "    - conda  : Test 'conda' targets"
    echo "    - lint   : Test 'linting' targets"
    echo "    - poetry : Test 'poetry' targets"
    echo "    - poetry-pipx : Test 'poetry' targets related to pipx"
    echo "    - uv : Test 'uv' targets"
    echo "    - uv-pipx : Test 'uv' targets related to pipx"
    echo "    - test   : Test 'test' targets"

    echo
    echo " Full test suite"
    echo
    echo "    - all    : Run most tests, except 'poetry-pipx'"
    echo

}

if [[ "$#" -eq 0 ]]; then
    list
fi

for var in "$@"
do
    # Order is set according to use, not alphabetical order
    case "$var" in
        "list")
            list
            ;;
        "base")
            base
            ;;
        "conda")
            conda
            ;;
        "lint")
            lint
            ;;
        "poetry")
            poetry
            ;;
        "poetry-pipx")
            poetry-pipx
            ;;
        "uv")
            uv
            ;;
        "uv-pipx")
            uv-pipx
            ;;
        "all")
            all
            ;;
        *)
            echo "* * * * * * * * * * * * * * * * * * * * * * * * * "
            echo "* ""$var"" is not a valid input "
            echo "* Use the list command to see available inputs"
            echo "* * * * * * * * * * * * * * * * * * * * * * * * *"
            echo
            list
            echo
            exit 1
    esac
done