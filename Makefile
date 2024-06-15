fastapi:
	dotenv -f .env run -- poetry run fastapi dev api

test:
	poetry run python -m unittest discover tests/

smoke:
	poetry run python smoketest.py