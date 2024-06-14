fastapi:
	fastapi dev api

test:
	python -m unittest discover tests/

smoke:
	python smoketest.py