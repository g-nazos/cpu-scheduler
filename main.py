#!/usr/bin/env python3
"""
Distributed CPU Scheduling via Ascending Auctions

Main entry point for running experiments based on Section 2.3.3
of "Multiagent Systems" by Shoham & Leyton-Brown.

Usage:
    python main.py                     # Run all book examples
    python main.py --example 1         # Run book example 1 (8-slot)
    python main.py --example 2         # Run book example 2 (no equilibrium)
    python main.py --example 3         # Run book example 3 (suboptimal)
    python main.py --sensitivity       # Run epsilon sensitivity analysis
    python main.py --all               # Run all experiments with visualizations
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.models.market import Market
from src.auction.ascending import AscendingAuction
from src.auction.equilibrium import check_equilibrium, print_equilibrium_report
from src.experiments.scenarios import (
    create_book_example_1,
    create_book_example_2,
    create_book_example_3,
)
from src.experiments.metrics import (
    compute_metrics,
    print_metrics_report,
    run_epsilon_sensitivity,
)
from src.visualization.plots import (
    plot_price_evolution,
    plot_allocation_timeline,
    plot_epsilon_sensitivity,
    plot_convergence_trace,
)

import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_book_example_1(epsilon: float = 0.25, show_trace: bool = True) -> None:
    """
    Run the main 8-slot processor example from Section 2.3.3.
    
    Expected behavior:
    - With ε=0.25: Converges to equilibrium in ~24 rounds
    - With ε=1.00: Does NOT reach equilibrium
    """
    print("\n" + "=" * 70)
    
    market = create_book_example_1()
    print(f"\nInitial Market Configuration:")
    print(f"  Slots: {len(market.slots)} (9am-5pm), Reserve: $3.00/hour")
    print(f"  Agents:")
    for agent in market.agents:
        print(f"    {agent.name}: λ={agent.required_slots}, d=slot_{agent.deadline_slot_id}, w=${agent.worth:.2f}")
    
    # Run auction
    print(f"\nRunning Ascending Auction with ε=${epsilon:.2f}...")
    auction = AscendingAuction(epsilon=epsilon)
    result = auction.run(market)
    
    if show_trace:
        result.print_trace(max_rounds=30)
    
    # Check equilibrium
    eq_result = check_equilibrium(result.market)
    print_equilibrium_report(eq_result, result.market)
    
    # Compute and print metrics
    metrics = compute_metrics(result, epsilon)
    print_metrics_report(metrics)
    
    return result, metrics


def run_book_example_2() -> None:
    """
    Run the "no equilibrium" example from Table 2.1.
    
    This demonstrates that competitive equilibrium may NOT exist
    due to complementarity in valuations.
    """
    print("\n" + "=" * 70)
    
    market = create_book_example_2()
    print(f"\nMarket Configuration:")
    print(f"  Slots: 2 (9am, 10am), Reserve: $3.00/hour")
    print(f"  Agents:")
    for agent in market.agents:
        print(f"    {agent.name}: λ={agent.required_slots}, d=slot_{agent.deadline_slot_id}, w=${agent.worth:.2f}")
    
    # Run auction with different epsilons
    for eps in [0.25, 0.5, 1.0]:
        print(f"\nRunning auction with ε=${eps:.2f}...")
        auction = AscendingAuction(epsilon=eps, max_iterations=100)
        result = auction.run(market)
        
        eq_result = check_equilibrium(result.market)
        print(f"  Iterations: {result.iterations}")
        print(f"  Solution: ${result.final_solution_value:.2f}")
        print(f"  Is Equilibrium: {eq_result.is_equilibrium}")
        
        if not eq_result.is_equilibrium and eq_result.violations:
            print(f"  Violations: {eq_result.violations[0]}")


def run_book_example_3() -> None:
    """
    Run the "arbitrarily suboptimal" example.
    
    This demonstrates that the ascending auction can produce
    allocations arbitrarily far from optimal.
    """
    print("\n" + "=" * 70)
    
    market = create_book_example_3()
    print(f"\nMarket Configuration:")
    print(f"  Slot 0: 9am, Reserve: $1.00")
    print(f"  Slot 1: 10am, Reserve: $9.00")
    print(f"  Agents:")
    for agent in market.agents:
        print(f"    {agent.name}: λ={agent.required_slots}, d=slot_{agent.deadline_slot_id}, w=${agent.worth:.2f}")
    
    # Run auction
    print(f"\nRunning Ascending Auction with ε=$0.25...")
    auction = AscendingAuction(epsilon=0.25)
    result = auction.run(market)
    
    print(f"\nAuction Result:")
    print(f"  Solution: ${result.final_solution_value:.2f}")
    print(f"  Allocation:")
    for agent in market.agents:
        slots = result.market.allocations.get(agent.agent_id, frozenset())
        slot_ids = sorted([s.slot_id for s in slots])
        if slot_ids:
            print(f"    {agent.name}: slots {slot_ids}")
    print("\n  Note: By adjusting values, auction outcome can be made arbitrarily far from optimal.")


def run_epsilon_sensitivity_analysis(show_plots: bool = True) -> None:
    """
    Run epsilon sensitivity analysis on Book Example 1.
    """
    print("\n" + "=" * 70)
    print("EPSILON SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    market = create_book_example_1()
    epsilons = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    
    print(f"\nTesting epsilon values: {epsilons}")
    print("Running experiments...")
    
    sensitivity = run_epsilon_sensitivity(market, epsilons)
    
    print("\nResults:")
    print("-" * 50)
    print(f"{'Epsilon':<10} {'Iterations':<12} {'Equilibrium'}")
    print("-" * 50)
    for i, eps in enumerate(sensitivity.epsilons):
        eq_str = "YES" if sensitivity.equilibrium_achieved[i] else "NO"
        print(f"{eps:<10.2f} {sensitivity.iterations[i]:<12} {eq_str}")
    print("-" * 50)
    
    if show_plots:
        fig = plot_epsilon_sensitivity(sensitivity, 
                                       title="Epsilon Sensitivity - Book Example 1")
        plt.show()


def run_all_experiments(save_plots: bool = False) -> None:
    """Run all experiments with full visualization."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Example 1 with ε=0.25 (should converge)
    print("\n" + "#" * 70)
    print("# Running Book Example 1 with ε=0.25")
    print("#" * 70)
    result1, metrics1 = run_book_example_1(epsilon=0.25, show_trace=True)
    
    if save_plots:
        plot_price_evolution(result1, title="Price Evolution - Example 1 (ε=0.25)",
                            save_path=str(output_dir / "ex1_prices_025.png"))
        plot_allocation_timeline(result1.market, title="Allocation - Example 1 (ε=0.25)",
                                save_path=str(output_dir / "ex1_allocation_025.png"))
    
    # Example 1 with ε=1.0 (should NOT converge to equilibrium)
    print("\n" + "#" * 70)
    print("# Running Book Example 1 with ε=1.0")
    print("#" * 70)
    result1b, metrics1b = run_book_example_1(epsilon=1.0, show_trace=True)
    
    # Example 2 (no equilibrium)
    print("\n" + "#" * 70)
    print("# Running Book Example 2 (No Equilibrium Case)")
    print("#" * 70)
    run_book_example_2()
    
    # Example 3 (suboptimal)
    print("\n" + "#" * 70)
    print("# Running Book Example 3 (Suboptimal Case)")
    print("#" * 70)
    run_book_example_3()
    
    # Epsilon sensitivity
    print("\n" + "#" * 70)
    print("# Running Epsilon Sensitivity Analysis")
    print("#" * 70)
    run_epsilon_sensitivity_analysis(show_plots=not save_plots)
    
    if save_plots:
        market = create_book_example_1()
        sensitivity = run_epsilon_sensitivity(market, [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        plot_epsilon_sensitivity(sensitivity, 
                                save_path=str(output_dir / "epsilon_sensitivity.png"))
        print(f"\nPlots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Distributed CPU Scheduling via Ascending Auctions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     Run all book examples
  python main.py --example 1         Run 8-slot processor example
  python main.py --example 1 --eps 1.0   Run example 1 with epsilon=1.0
  python main.py --sensitivity       Run epsilon sensitivity analysis
  python main.py --all --save        Run all experiments and save plots
        """
    )
    
    parser.add_argument("--example", "-e", type=int, choices=[1, 2, 3],
                       help="Run specific book example (1, 2, or 3)")
    parser.add_argument("--eps", type=float, default=0.25,
                       help="Epsilon value for auction (default: 0.25)")
    parser.add_argument("--sensitivity", "-s", action="store_true",
                       help="Run epsilon sensitivity analysis")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Run all experiments")
    parser.add_argument("--save", action="store_true",
                       help="Save plots to output directory")
    parser.add_argument("--no-plots", action="store_true",
                       help="Disable plot display")
    
    args = parser.parse_args()
    
    # Default behavior: run all book examples
    if not any([args.example, args.sensitivity, args.all]):
        args.all = True
    
    try:
        if args.all:
            run_all_experiments(save_plots=args.save)
        elif args.example == 1:
            run_book_example_1(epsilon=args.eps)
            if not args.no_plots:
                plt.show()
        elif args.example == 2:
            run_book_example_2()
        elif args.example == 3:
            run_book_example_3()
        elif args.sensitivity:
            run_epsilon_sensitivity_analysis(show_plots=not args.no_plots)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
