"""
Retrieval Strategy Comparison Script (Part 2)
=============================================
Runs only the multi-query based strategies.
"""

from custom.benchmarks.suites.strategy_comparison import (
    STRATEGIES_SECOND_RUN,
    run_strategy_suite,
)


def main():
    run_strategy_suite(
        selected_strategies=STRATEGIES_SECOND_RUN,
        title="RETRIEVAL STRATEGY COMPARISON (PART 2: MULTI-QUERY)",
        output_filename="variations_reranker_results_part2.txt",
    )


if __name__ == "__main__":
    main()
