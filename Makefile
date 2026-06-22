.PHONY: test check

test:
	python -m unittest discover -s tests

check: test
	python -m compileall -q src scripts tests

