"""
Metrics for evaluating auction performance.

Computes solution value, iterations, and other metrics for experimental analysis.
"""

from dataclasses import dataclass

from src.models.market import Market
from src.auction.ascending import AuctionResult


@dataclass
class ExperimentMetrics:
    """Metrics computed from an auction experiment."""
    auction_solution_value: float
    iterations: int
    num_rounds: int
    converged: bool
    epsilon: float
    num_agents: int
    num_slots: int


def compute_metrics(result: AuctionResult, epsilon: float) -> ExperimentMetrics:
    """
    Compute metrics for an auction result.

    Args:
        result: The auction result
        epsilon: The epsilon used in the auction

    Returns:
        ExperimentMetrics with computed values
    """
    market = result.market
    return ExperimentMetrics(
        auction_solution_value=result.final_solution_value,
        iterations=result.iterations,
        num_rounds=len(result.rounds),
        converged=result.converged,
        epsilon=epsilon,
        num_agents=len(market.agents),
        num_slots=len(market.slots)
    )


def print_metrics_report(metrics: ExperimentMetrics) -> None:
    """Print a formatted metrics report."""
    print("\n" + "=" * 60)
    print("EXPERIMENT METRICS")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Agents: {metrics.num_agents}")
    print(f"  Slots: {metrics.num_slots}")
    print(f"  Epsilon: {metrics.epsilon}")
    print(f"\nAuction Performance:")
    print(f"  Converged: {metrics.converged}")
    print(f"  Iterations: {metrics.iterations}")
    print(f"  Rounds: {metrics.num_rounds}")
    print(f"\nSolution Value:")
    print(f"  Auction: ${metrics.auction_solution_value:.2f}")
    print("=" * 60)


@dataclass
class EpsilonSensitivityResult:
    """Result of epsilon sensitivity analysis."""
    epsilons: list[float]
    iterations: list[int]


def run_epsilon_sensitivity(
    market: Market,
    epsilons: list[float],
    max_iterations: int = 10000
) -> EpsilonSensitivityResult:
    """
    Run sensitivity analysis varying epsilon.

    Args:
        market: Market to test
        epsilons: List of epsilon values to test
        max_iterations: Max iterations per run

    Returns:
        EpsilonSensitivityResult with results for each epsilon
    """
    from src.auction.ascending import AscendingAuction

    iterations = []
    for eps in epsilons:
        auction = AscendingAuction(epsilon=eps, max_iterations=max_iterations)
        result = auction.run(market)
        iterations.append(result.iterations)
    return EpsilonSensitivityResult(epsilons=epsilons, iterations=iterations)
