"""Backend-free request bounds and MLX allocator measurement orchestration."""

from time import perf_counter


PEAK_MLX_SCOPE = (
    "MLX allocator peak bytes since reset immediately before each measured request; "
    "after model load and warmup, including resident model allocation; "
    "not a baseline-subtracted increment or physical RAM. Synchronization, reset "
    "and peak query are outside request clocks."
)


REQUEST_ELAPSED_SCOPE = (
    "sum of full request wall times, including cache/input setup and release; "
    "excluding allocator instrumentation and progress reporting"
)


def request_cache_capacity(loader, checkpoint, request):
    """Apply the taught prompt-plus-output bound only to bounded Week 2."""
    if loader == "week2" and (checkpoint or "selected") != "kv-cache":
        return len(request.prompt_token_ids) + request.max_new_tokens
    return None


def measure_request(run, request, *, memory, clock=perf_counter):
    """Measure one full request, excluding allocator instrumentation from time."""
    memory.synchronize()
    memory.reset_peak_memory()
    started = clock()
    result = run(request)
    elapsed = clock() - started
    memory.synchronize()
    peak_bytes = memory.get_peak_memory()
    return result, elapsed, peak_bytes
