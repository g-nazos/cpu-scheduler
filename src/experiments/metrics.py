"""
Metrics for evaluating auction performance.

Computes solution value, equilibrium status, and other
metrics for experimental analysis.
"""

from dataclasses import dataclass

from src.models.market import Market
from src.auction.ascending import AuctionResult
from src.auction.equilibrium import EquilibriumChecker, EquilibriumCheckResult


@dataclass
class ExperimentMetrics:
    """Metrics computed from an auction experiment."""
    auction_solution_value: float
    iterations: int
    num_rounds: int
    converged: bool
    is_equilibrium: bool
    equilibrium_result: EquilibriumCheckResult
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
    checker = EquilibriumChecker(market)
    eq_result = checker.check()

    return ExperimentMetrics(
        auction_solution_value=result.final_solution_value,
        iterations=result.iterations,
        num_rounds=len(result.rounds),
        converged=result.converged,
        is_equilibrium=eq_result.is_equilibrium,
        equilibrium_result=eq_result,
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

    print(f"\nEquilibrium:")
    print(f"  Is Equilibrium: {metrics.is_equilibrium}")
    if not metrics.is_equilibrium and metrics.equilibrium_result.violations:
        print("  Violations:")
        for v in metrics.equilibrium_result.violations[:3]:
            print(f"    - {v}")

    print("=" * 60)


@dataclass
class EpsilonSensitivityResult:
    """Result of epsilon sensitivity analysis."""
    epsilons: list[float]
    iterations: list[int]
    equilibrium_achieved: list[bool]


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
    equilibrium_achieved = []

    for eps in epsilons:
        auction = AscendingAuction(epsilon=eps, max_iterations=max_iterations)
        result = auction.run(market)
        checker = EquilibriumChecker(result.market)
        eq_result = checker.check()
        iterations.append(result.iterations)
        equilibrium_achieved.append(eq_result.is_equilibrium)

    return EpsilonSensitivityResult(
        epsilons=epsilons,
        iterations=iterations,
        equilibrium_achieved=equilibrium_achieved
    )
