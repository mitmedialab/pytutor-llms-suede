import os

import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("CLAUDE_API_KEY", "test-key")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


def pytest_addoption(parser: pytest.Parser) -> None:
	parser.addoption(
		"--run-provider-benchmarks",
		action="store_true",
		default=False,
		help="Run live provider benchmark tests (requires real API keys).",
	)
