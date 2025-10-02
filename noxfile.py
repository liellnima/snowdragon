import re
from pathlib import Path

import nox

ARG_RE = re.compile(r"^-[-\w=]+$")  # e.g. "-k", "--maxfail=1", "tests/foo.py"

nox.options.reuse_existing_virtualenvs = True  # Reuse virtual environments
nox.options.sessions = ["precommit"]


def get_paths(session):
    package_path = Path(session.bin).parent.parent.parent
    main_package = package_path / "src"
    tests = package_path / "tests"
    scripts = package_path / "scripts"
    return {
        "all": [
            main_package,
            tests,
            scripts,
        ],
        "module": [
            main_package,
            scripts,
        ],
        "root": [package_path],
    }


#
# Sessions
#
@nox.session()
def pylint(session):
    paths = get_paths(session)
    session.run("pylint", *paths["module"], external=True)


@nox.session()
def flake8(session):
    paths = get_paths(session)
    session.run("flake8", *paths["all"], external=True)


@nox.session()
def complexity(session):
    paths = get_paths(session)
    session.run("flake8", "--max-complexity", "7", *paths["all"], external=True)


@nox.session()
def docformatter(session):
    paths = get_paths(session)
    session.run(
        "docformatter",
        "--config",
        f"{paths['all'][0].parent}/pyproject.toml",
        *paths["all"],
        external=True,
    )


@nox.session()
def check(session):
    paths = get_paths(session)
    session.run("black", "--check", *paths["all"], external=True)
    session.run("isort", *paths["all"], "--check", external=True)
    session.run("flynt", *paths["all"], external=True)
    session.run(
        "docformatter",
        "--config",
        f"{paths['all'][0].parent}/pyproject.toml",
        *paths["all"],
        external=True,
    )
    session.run("flake8", *paths["all"], external=True)
    session.run("pylint", *paths["module"], external=True)


@nox.session()
def fix(session):
    paths = get_paths(session)
    session.run("autoflake", "-v", *paths["all"], external=True)
    session.run("autopep8", *paths["all"], external=True)
    session.run("black", *paths["all"], external=True)
    session.run("isort", *paths["all"], external=True)
    session.run("flynt", *paths["all"], external=True)
    session.run(
        "docformatter",
        "--in-place",
        "--config",
        f"{paths['all'][0].parent}/pyproject.toml",
        *paths["all"],
        external=True,
    )
    session.run("mdformat", *paths["root"], external=True)


@nox.session()
def precommit(session):
    session.run("pre-commit", "run", "--all-files", external=True)


@nox.session()
def autoflake(session):
    paths = get_paths(session)
    session.run("autoflake", "-v", *paths["all"], external=True)


@nox.session()
def autopep(session):
    paths = get_paths(session)
    session.run("autopep8", *paths["all"], external=True)


@nox.session()
def black(session):
    paths = get_paths(session)
    session.run("black", "--check", *paths["all"], external=True)


@nox.session()
def isort(session):
    paths = get_paths(session)
    session.run("isort", *paths["all"], "--check", external=True)


@nox.session()
def flynt(session):
    paths = get_paths(session)
    session.run("flynt", *paths["all"], external=True)


@nox.session()
def mdformat(session):
    paths = get_paths(session)
    session.run("mdformat", *paths["root"], external=True)


@nox.session(name="ruff-lint")
def ruff_lint(session):
    paths = get_paths(session)
    session.run("ruff", "check", *paths["all"], external=True)


@nox.session(name="ruff-fix")
def ruff_fix(session):
    paths = get_paths(session)
    session.run("ruff", "check", "--fix", *paths["all"], external=True)


@nox.session(name="ruff-format")
def ruff_format(session):
    paths = get_paths(session)
    session.run("ruff", "format", *paths["all"], external=True)


@nox.session()
def test(session):
    session.run("pytest", external=True)


@nox.session()
def test_custom(session):
    for a in session.posargs:
        if not ARG_RE.match(a):
            session.error(f"unsafe pytest argument detected: {a!r}")

    session.run(
        "python", "-m", "pytest", external=True, *session.posargs
    )  # Pass additional arguments directly to pytest


@nox.session()
def test_nb(session):
    session.run(
        "pytest",
        "--nbval",
        "tests/test_notebooks/",
        "--nbval-sanitize-with=tests/test_notebooks/sanitize_file.cfg",
        external=True,
    )
