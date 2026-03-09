# Testing Strategy

This library is best tested as a layered suite so fast unit tests catch most regressions, while provider integration tests validate real API compatibility when needed.

## 1) Test Layers

- **Unit tests (default in CI):**
  - Pure utilities (`compute_delta`, schema->model generation)
  - Message normalization (`msg_content`)
  - Provider selection/fallback behavior (`Provider._Select`, stream wrappers)
  - Provider acceptance rules and chunk parsing logic (mocked producers)
- **Contract tests (default in CI):**
  - Every provider must satisfy the same stream contract:
    - emits `chunk` events with monotonic index
    - preserves accumulated text semantics
    - emits `error` event on producer exceptions
- **Live integration tests (optional/manual):**
  - Run against real OpenAI/Google/Anthropic/OpenRouter APIs using env keys
  - Validate end-to-end streaming shape and auth/model compatibility

## 2) Recommended Libraries

- `pytest`: base framework and fixtures
- `pytest-asyncio`: async test support for stream APIs
- `pytest-cov`: coverage reporting and CI thresholds

Optional additions as the project grows:

- `hypothesis`: property-based tests for `compute_delta` and message normalization edge cases
- `pytest-xdist`: parallel test execution in CI
- `vcrpy` or `respx`: stable HTTP-recorded tests for non-streaming provider calls

## 3) What Exists Now

Current suite covers:

- Utility behavior in `tests/test_utils.py`
- Message normalization in `tests/test_msg_content.py`
- Base provider/stream semantics in `tests/test_provider_base.py`
- Provider-specific model routing and stream startup behavior in `tests/test_providers_specific.py`

The tests run without real API keys by setting dummy credentials in `tests/conftest.py`.

## 4) Running Tests

```bash
/workspaces/pytutor-llms-suede/.venv/bin/python -m pytest -q
```

Using the helper script:

```bash
./scripts/run-tests.sh
./scripts/run-tests.sh tests/test_provider_base.py
./scripts/run-tests.sh tests/test_provider_base.py::test_text_stream_collect_collects_all_events
```

With coverage:

```bash
/workspaces/pytutor-llms-suede/.venv/bin/python -m pytest --cov=release --cov-report=term-missing
```

## 5) Provider Benchmarks (opt-in)

Provider benchmark tests live in `tests/test_provider_benchmarks.py` and are skipped by default.

To run them, pass the explicit opt-in flag:

```bash
./scripts/run-tests.sh tests/test_provider_benchmarks.py --run-provider-benchmarks -s
```

Required env vars (real keys):

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `CLAUDE_API_KEY`
- `OPENROUTER_API_KEY`

Optional model overrides:

- `BENCHMARK_OPENAI_MODEL` (default `gpt-4o-mini`)
- `BENCHMARK_GOOGLE_MODEL` (default `gemini-2.0-flash`)
- `BENCHMARK_ANTHROPIC_MODEL` (default `claude-3-5-haiku-latest`)
- `BENCHMARK_OPENROUTER_MODEL` (default `openrouter/openai/gpt-4o-mini`)

## 6) CI Recommendations

Use this minimum quality gate for future development:

- run `pytest` on every PR
- fail builds on test failures
- start with coverage floor at 85% and raise over time
- keep integration tests in a separate workflow requiring provider secrets

## 7) Adding New Providers

For each new provider:

1. Add provider-specific tests for model prefix acceptance/rejection.
2. Add text chunk parser tests for empty/non-text chunk handling.
3. Reuse base stream contract tests to verify `chunk` and `error` semantics.
4. Add one optional live integration test for a known model id.
