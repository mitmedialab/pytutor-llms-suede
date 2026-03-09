from dataclasses import dataclass
import os
from time import perf_counter
from typing import TypedDict

import pytest

from release import Provider
from release.providers.anthropic import AnthropicProvider
from release.providers.google import GoogleProvider
from release.providers.openai import OpenAIProvider
from release.providers.openrouter import OpenRouterProvider


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    provider: Provider
    model_env: str
    model_default: str
    key_env: str


class BenchmarkMetrics(TypedDict):
    provider: str
    model: str
    ttfc_s: float
    total_s: float
    chunks: int
    chars: int
    chunks_per_s: float
    chars_per_s: float


CASES = [
    BenchmarkCase(
        name="openai",
        provider=OpenAIProvider(),
        model_env="BENCHMARK_OPENAI_MODEL",
        model_default="gpt-4o-mini",
        key_env="OPENAI_API_KEY",
    ),
    BenchmarkCase(
        name="google",
        provider=GoogleProvider(),
        model_env="BENCHMARK_GOOGLE_MODEL",
        model_default="gemini-2.0-flash",
        key_env="GEMINI_API_KEY",
    ),
    BenchmarkCase(
        name="anthropic",
        provider=AnthropicProvider(),
        model_env="BENCHMARK_ANTHROPIC_MODEL",
        model_default="claude-3-5-haiku-latest",
        key_env="CLAUDE_API_KEY",
    ),
    BenchmarkCase(
        name="openrouter",
        provider=OpenRouterProvider(),
        model_env="BENCHMARK_OPENROUTER_MODEL",
        model_default="openrouter/openai/gpt-4o-mini",
        key_env="OPENROUTER_API_KEY",
    ),
]


def _real_key_present(key_name: str) -> bool:
    key = os.getenv(key_name, "")
    return bool(key) and key != "test-key"


async def _measure_text_stream(case: BenchmarkCase, model: str) -> BenchmarkMetrics:
    request = Provider.TextStream.Request(
        model=model,
        messages=[
            {"role": "system", "content": "You are concise."},
            {
                "role": "user",
                "content": "Write 120 characters describing how latency benchmarking works.",
            },
        ],
    )

    start = perf_counter()
    first_chunk_at: float | None = None
    chunk_count = 0
    char_count = 0

    stream = await Provider.TextStream.Select(request, case.provider)

    async for event in stream:
        if event.type == "error":
            raise event.payload

        if first_chunk_at is None:
            first_chunk_at = perf_counter()

        chunk_count += 1
        char_count += len(event.payload.delta)

    end = perf_counter()

    if first_chunk_at is None:
        raise AssertionError("No chunks received from provider stream")

    total_s = max(end - start, 1e-9)
    ttfc_s = max(first_chunk_at - start, 0.0)

    return {
        "provider": case.name,
        "model": model,
        "ttfc_s": ttfc_s,
        "total_s": total_s,
        "chunks": chunk_count,
        "chars": char_count,
        "chunks_per_s": chunk_count / total_s,
        "chars_per_s": char_count / total_s,
    }


@pytest.mark.asyncio
@pytest.mark.benchmark
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
async def test_provider_text_stream_benchmark(
    case: BenchmarkCase, request: pytest.FixtureRequest
) -> None:
    if not request.config.getoption("--run-provider-benchmarks"):
        pytest.skip("Use --run-provider-benchmarks to enable live benchmark tests")

    if not _real_key_present(case.key_env):
        pytest.skip(f"Missing real API key in {case.key_env}")

    model = os.getenv(case.model_env, case.model_default)
    metrics = await _measure_text_stream(case, model)

    print(
        "\n"
        f"[{metrics['provider']}] model={metrics['model']} "
        f"ttfc={metrics['ttfc_s']:.3f}s "
        f"total={metrics['total_s']:.3f}s "
        f"chunks={metrics['chunks']} "
        f"chars={metrics['chars']} "
        f"chunks/s={metrics['chunks_per_s']:.2f} "
        f"chars/s={metrics['chars_per_s']:.2f}"
    )

    assert metrics["ttfc_s"] >= 0
    assert metrics["total_s"] >= metrics["ttfc_s"]
    assert metrics["chunks"] > 0
    assert metrics["chars"] > 0
