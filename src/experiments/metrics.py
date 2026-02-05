"""
Metrics for evaluating auction performance.

Computes welfare, optimality gap, equilibrium status, and other
metrics for experimental analysis.
"""

from dataclasses import dataclass
from typing import Optional

from src.models.market import Market
from src.auction.ascending import AuctionResult
from src.auction.equilibrium import EquilibriumChecker, EquilibriumCheckResult
from src.optimization.integer_program import IntegerProgramSolver, IPSolution


@dataclass
class ExperimentMetrics:
    """Metrics computed from an auction experiment."""
    # Auction performance
    auction_welfare: float
    iterations: int
    num_rounds: int
    converged: bool
    
    # Optimality
    optimal_welfare: float
    welfare_gap: float
    welfare_ratio: float  # auction_welfare / optimal_welfare
    
    # Equilibrium
    is_equilibrium: bool
    equilibrium_result: EquilibriumCheckResult
    
    # Configuration
    epsilon: float
    num_agents: int
    num_slots: int


def compute_metrics(
    result: AuctionResult,
    epsilon: float,
    optimal_solution: Optional[IPSolution] = None
) -> ExperimentMetrics:
    """
    Compute comprehensive metrics for an auction result.
    
    Args:
        result: The auction result
        epsilon: The epsilon used in the auction
        optimal_solution: Pre-computed optimal solution (computed if None)
        
    Returns:
        ExperimentMetrics with all computed values
    """
    market = result.market
    
    # Compute optimal if not provided
    if optimal_solution is None:
        solver = IntegerProgramSolver(market)
        optimal_solution = solver.solve()
    
    # Equilibrium check
    checker = EquilibriumChecker(market)
    eq_result = checker.check()
    
    # Compute gaps
    auction_welfare = result.final_welfare
    optimal_welfare = optimal_solution.optimal_welfare
    welfare_gap = optimal_welfare - auction_welfare
    welfare_ratio = auction_welfare / optimal_welfare if optimal_welfare > 0 else 0.0
    
    return ExperimentMetrics(
        auction_welfare=auction_welfare,
        iterations=result.iterations,
        num_rounds=len(result.rounds),
        converged=result.converged,
        optimal_welfare=optimal_welfare,
        welfare_gap=welfare_gap,
        welfare_ratio=welfare_ratio,
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
    
    print(f"\nWelfare:")
    print(f"  Auction Welfare: ${metrics.auction_welfare:.2f}")
    print(f"  Optimal Welfare: ${metrics.optimal_welfare:.2f}")
    print(f"  Gap: ${metrics.welfare_gap:.2f}")
    print(f"  Ratio: {metrics.welfare_ratio:.2%}")
    
    print(f"\nEquilibrium:")
    print(f"  Is Equilibrium: {metrics.is_equilibrium}")
    if not metrics.is_equilibrium and metrics.equilibrium_result.violations:
        print("  Violations:")
        for v in metrics.equilibrium_result.violations[:3]:  # Show first 3
            print(f"    - {v}")
    
    print("=" * 60)


@dataclass
class EpsilonSensitivityResult:
    """Result of epsilon sensitivity analysis."""
    epsilons: list[float]
    iterations: list[int]
    welfare_gaps: list[float]
    welfare_ratios: list[float]
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
    
    # Get optimal solution once
    solver = IntegerProgramSolver(market)
    optimal = solver.solve()
    
    iterations = []
    welfare_gaps = []
    welfare_ratios = []
    equilibrium_achieved = []
    
    for eps in epsilons:
        # Run auction
        auction = AscendingAuction(epsilon=eps, max_iterations=max_iterations)
        result = auction.run(market)
        
        # Compute metrics
        checker = EquilibriumChecker(result.market)
        eq_result = checker.check()
        
        gap = optimal.optimal_welfare - result.final_welfare
        ratio = result.final_welfare / optimal.optimal_welfare if optimal.optimal_welfare > 0 else 0
        
        iterations.append(result.iterations)
        welfare_gaps.append(gap)
        welfare_ratios.append(ratio)
        equilibrium_achieved.append(eq_result.is_equilibrium)
    
    return EpsilonSensitivityResult(
        epsilons=epsilons,
        iterations=iterations,
        welfare_gaps=welfare_gaps,
        welfare_ratios=welfare_ratios,
        equilibrium_achieved=equilibrium_achieved
    )
