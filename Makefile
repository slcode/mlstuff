.PHONY: install test mypy

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

mypy:
	mypy .

test:
	pytest