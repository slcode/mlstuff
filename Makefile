.PHONY: install test mypy

local_test: mypy test

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

mypy:
	mypy .

test:
	pytest