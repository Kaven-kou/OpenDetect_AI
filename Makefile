.PHONY: help install dev run test integration-tests lint format eval intent-eval clarify-eval route-eval

help:
	@echo 'Targets:'
	@echo '  install             Sync runtime dependencies with uv'
	@echo '  dev                 Sync project + dev dependencies with uv'
	@echo '  run                 Start the local LangGraph dev server'
	@echo '  test                Run unit tests'
	@echo '  integration-tests   Run integration tests'
	@echo '  eval                Run RAG retrieval eval (baseline vs pipeline)'
	@echo '  intent-eval         Run Search intent golden set (live model)'
	@echo '  clarify-eval        Run clarification judgment golden set (live model)'
	@echo '  route-eval          Run Supervisor routing golden set (live model)'
	@echo '  lint                Run Ruff checks'
	@echo '  format              Format with Ruff'

install:
	uv sync --no-dev

dev:
	uv sync

run:
	uv run langgraph dev

test:
	uv run python -m pytest tests/unit_tests -q

integration-tests:
	uv run python -m pytest tests/integration_tests -q

eval:
	uv run python -m opendetect_ai.eval.rag_eval

intent-eval:
	uv run python -m opendetect_ai.eval.intent_eval

clarify-eval:
	uv run python -m opendetect_ai.eval.clarify_eval

route-eval:
	uv run python -m opendetect_ai.eval.route_eval

lint:
	uv run python -m ruff check src tests

format:
	uv run python -m ruff format src tests
