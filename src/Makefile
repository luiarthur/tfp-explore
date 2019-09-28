SHELL = /bin/bash

.PHONY: activate deactivate install test init clean write-freeze console

VENV = venv-tfp

ACTIVATE = . $(VENV)/bin/activate

activate:
	$(ACTIVATE) && bash

deactivate:
	@echo "Execute:"
	@echo "deactivate"
	@echo

init-env:
	python3 -m venv $(VENV)  # completely independent installations
# python3 -m venv --system-site-packages $(VENV)  # uses local libs if available 
	$(ACTIVATE) && pip install --upgrade pip

init: init-env install

console:
	$(ACTIVATE) && python3

clean:
	rm -rf $(VENV)/

freeze:
	@$(ACTIVATE) && python3 -m pip freeze

install: requirements.txt
	$(ACTIVATE) && pip install -Ur requirements.txt

nb:
	$(ACTIVATE) && jupyter notebook
