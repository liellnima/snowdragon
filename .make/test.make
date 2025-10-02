## -- Tests targets ------------------------------------------------------------------------------------------------- ##

.PHONY: test
test: ## Run all tests
	$(ENV_COMMAND_TOOL) nox -s test

TEST_ARGS ?=
MARKER_TEST_ARGS = -m "$(TEST_ARGS)"
SPECIFIC_TEST_ARGS = -k "$(TEST_ARGS)"
CUSTOM_TEST_ARGS = "$(TEST_ARGS)"

.PHONY: test-marker
test-marker: ## Run tests using pytest markers. Ex. make test-marker TEST_ARGS="<marker>"
	@if [ -n "$(TEST_ARGS)" ]; then \
		$(ENV_COMMAND_TOOL) nox -s test_custom -- -- $(MARKER_TEST_ARGS); \
	else \
		echo "" ; \
    	echo 'ERROR : Variable TEST_ARGS has not been set, please rerun the command like so :' ; \
	  	echo "" ; \
    	echo '            make test-marker TEST_ARGS="<marker>"' ; \
	  	echo "" ; \
    fi
.PHONY: test-specific
test-specific: ## Run specific tests using the -k option. Ex. make test-specific TEST_ARGS="<name-of-test>"
	@if [ -n "$(TEST_ARGS)" ]; then \
  		$(ENV_COMMAND_TOOL) nox -s test_custom -- -- $(SPECIFIC_TEST_ARGS); \
	else \
		echo "" ; \
    	echo 'ERROR : Variable TEST_ARGS has not been set, please rerun the command like so :' ; \
	  	echo "" ; \
    	echo '            make test-specific TEST_ARGS="<name-of-the-test>"' ; \
	  	echo "" ; \
    fi

.PHONY: test-custom
test-custom: ## Run tests with custom args. Ex. make test-custom TEST_ARGS="-m 'not offline'"
	@if [ -n "$(TEST_ARGS)" ]; then \
  		$(ENV_COMMAND_TOOL) nox -s test_custom -- -- $(CUSTOM_TEST_ARGS); \
	else \
	  	echo "" ; \
    	echo 'ERROR : Variable TEST_ARGS has not been set, please rerun the command like so :' ; \
	  	echo "" ; \
    	echo '            make test-custom TEST_ARGS="<custom-args>"' ; \
	  	echo "" ; \
    fi