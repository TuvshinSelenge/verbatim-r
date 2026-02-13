import concurrent.futures


def run_with_timeout(func, timeout_sec: int):
    """Run a function in a thread with a hard timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            print(f"  TIMEOUT after {timeout_sec}s - skipping")
            raise TimeoutError(f"Timed out after {timeout_sec}s")


def is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection of provider/API rate-limit errors."""
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    text = str(exc).lower()
    return (
        " 429 " in f" {text} "
        or "error code: 429" in text
        or "rate limit" in text
        or "rate-limited" in text
    )
