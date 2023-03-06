#########
# LINTS #
#########
lint:  ## run static analysis with flake8
	python -m black --check multi_scale_expansion setup.py
	python -m flake8 multi_scale_expansion setup.py

# Alias
lints: lint

format:  ## run autoformatting with black
	python -m black multi_scale_expansion/ setup.py

# alias
fix: format

check:  ## check assets for packaging
	check-manifest -v

# Alias
checks: check

annotate:  ## run type checking
	python -m mypy ./multi_scale_expansion

#########
# TESTS #
#########
test: ## clean and run unit tests
	python -m pytest -v multi_scale_expansion/tests

coverage:  ## clean and run unit tests with coverage
	python -m pytest -v multi_scale_expansion/tests --cov=multi_scale_expansion --cov-branch --cov-fail-under=75 --cov-report term-missing

# Alias
tests: test