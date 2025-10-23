# Makefile documentation

This folder contains the base makefile targets.

It should not be modified outside the [Template's repository](https://github.com/RolnickLab/lab-advanced-template).

If there is a problem with the contents of these targets, please open an
issue [here](https://github.com/RolnickLab/lab-advanced-template/issues).

If you know how to fix the problem, please also consider opening a pull request with
your proposed solution.

You can always override the faulty targets located here by creating new targets
with the same names inside [Makefile.targets](../Makefile.targets).

The Makefile and `bump-my-version` related files in this directory are to help with
change tracking for the makefile itself. If you are using this makefile as part of a
template in another repository, you won't have to interact with them.

## Tests

Makefile tests can be found in the [.make/tests/](tests) folder. Current tests are
essentially bash scripts to test the different makefile targets.

These should only be run when modifying the makefiles inside the
[Template's repository](https://github.com/RolnickLab/lab-advanced-template).

They should never be run in a project implemented from the template, as they could cause
side effects to your project.
