# LLM Graphs
Library to plot directed graphs with LLMs

## Overview
This is an LLM library to create directed graphs from LLMs

You can run this as a script using the cli: 

```shell
python -m llm_graphs.knowledge_graph "The little Pricnce" --model gpt-3.5-turbo
```
This will generate a html file with the graph in the same directory. 

## Usage
This project is set up using poetry. To install the dependencies, run `poetry install` from the root of the project.

```shell
poetry install
```
To add a new dependency, run `poetry add <dependency>` from the root of the project.
```shell
poetry add <dependency>
```

### Pre-Commit Hooks
This project uses [pre-commit](https://pre-commit.com/) to run linting and formatting tools before each commit. To install the pre-commit hooks, run `pre-commit install` from the root of the project.

```shell
poetry run pre-commit install
```

To run the pre-commit hooks manually, run `pre-commit run --all-files` from the root of the project.

```shell
poetry run pre-commit run --all-files
```


### Testing
This project uses [pytest](https://docs.pytest.org/en/stable/) for testing. To run the tests, run `pytest` from the root of the project in the poetry shell.

```shell
poetry run pytest
```

There are sensible defaults for pytest setup in the `pyproject.toml` file. You can override these defaults by passing in command line arguments. For example, to run the tests with debug logging enabled, run `pytest --log-cli-level=DEBUG` from the root of the project.

```shell
poetry run pytest --log-cli-level=DEBUG
```

# Original cookie cutter template 
https://github.com/mikelane/cookiecutter-python-library/
