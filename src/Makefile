SHELL = /bin/bash

.PHONY: activate deactivate install test init clean write-freeze console

ACTIVATE = . venv/bin/activate

activate:
	$(ACTIVATE) && bash

deactivate:
	@echo "Execute:"
	@echo "deactivate"
	@echo

init:
	python3 -m venv --system-site-packages venv && \
		$(ACTIVATE) && python3 -m pip install -U pip

console:
	$(ACTIVATE) && python3

clean:
	rm -rf venv/

freeze:
	@$(ACTIVATE) && python3 -m pip freeze

install: requirements.txt
	$(ACTIVATE) && pip install -Ur requirements.txt

run-logistic-regression:
	$(ACTIVATE) && python3 tf2_logistic_regression.py

nb:
	$(ACTIVATE) && jupyter notebook
