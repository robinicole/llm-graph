fastapi:
	dotenv -f .env run -- poetry run fastapi dev api

test:
	poetry run python -m unittest discover tests/

smoke:
	python smoketest.py