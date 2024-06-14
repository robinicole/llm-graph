fastapi:
	fastapi dev api

test:
	python -m unittest discover

smoke:
	python smoketest.py