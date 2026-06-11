.PHONY: lint typecheck test all

lint:
	ruff check tin tests

typecheck:
	mypy tin --no-error-summary

test:
	NUMBA_DISABLE_JIT=1 pytest -v

all: lint typecheck test
