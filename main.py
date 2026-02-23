#!/usr/bin/env python3
"""
Usage:
    python main.py                     # Run all book examples
    python main.py --example 1         # Run book example 1 (8-slot)
    python main.py --example 2         # Run book example 2
    python main.py --example 3         # Run book example 3 (suboptimal)
    python main.py --example 4         # Run many-jobs example (8 jobs)
    python main.py --sensitivity       # Run epsilon sensitivity analysis
    python main.py --all               # Run all experiments with visualizations
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.models.market import Market
from src.auction.ascending import AscendingAuction, AuctionResult
from src.experiments.scenarios import (
    create_book_example_1,
    create_book_example_1_two_cpus,
    create_book_example_2,
    create_book_example_2_two_cpus,
    create_book_example_3,
    create_many_jobs_example,
    create_duplicate_example_1,
    create_competitive_scenario,
    create_24h_night_discount_scenario,
)
from src.experiments.metrics import (
    compute_metrics,
    print_metrics_report,
    run_epsilon_sensitivity,
    EpsilonSensitivityResult,
)
from src.visualization.plots import (
    plot_price_evolution,
    plot_allocation_timeline,
    plot_allocation_and_prices,
    plot_epsilon_sensitivity,
    plot_epsilon_sensitivity_all,
    plot_convergence_trace,
)

import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_final_market_state(result) -> None:
    """Print final allocation and bid prices after an auction run."""
    market = result.market
    print("  Final allocation:")
    for agent in market.agents:
        slots = market.allocations.get(agent.agent_id, frozenset())
        slot_ids = sorted([s.slot_id for s in slots])
        print(f"    {agent.name}: slots {slot_ids}")
    print("  Bid prices:")
    for slot in market.slots:
        price = market.bid_prices.get(slot.slot_id, slot.reserve_price)
        print(f"    Slot {slot.slot_id} ({slot.time_label}): ${price:.2f}")


def run_book_example_1(epsilon: float = 0.25, show_trace: bool = True, two_cpus: bool = False) -> None:
    """
    Run the main 8-slot processor example from Section 2.3.3.
    
    Expected behavior:
    - With ε=0.25: Converges in ~24 rounds
    - With ε=1.00: May take more rounds or not converge
    - With two_cpus=True: same jobs, 2 identical CPUs (16 slots).
    """
    print("\n" + "=" * 70)
    
    market = create_book_example_1_two_cpus() if two_cpus else create_book_example_1()
    print(f"\nInitial Market Configuration:")
    slots_desc = f"{len(market.slots)} (2 CPUs × 8 times, 9am-5pm)" if two_cpus else f"{len(market.slots)} (9am-5pm)"
    print(f"  Slots: {slots_desc}, Reserve: $3.00/hour")
    print(f"  Agents:")
    for agent in market.agents:
        print(f"    {agent.name}: λ={agent.required_slots}, d=slot_{agent.deadline_slot_id}, w=${agent.worth:.2f}")
    
    # Run auction
    print(f"\nRunning Ascending Auction with ε=${epsilon:.2f}...")
    auction = AscendingAuction(epsilon=epsilon)
    result = auction.run(market)
    
    if show_trace:
        result.print_trace(max_rounds=100)
    print("\nFinal market state:")
    print_final_market_state(result)
    # Compute and print metrics
    metrics = compute_metrics(result, epsilon)
    print_metrics_report(metrics)
    
    return result, metrics


def run_many_jobs_example(epsilon: float = 0.25, show_trace: bool = True, two_cpus: bool = False) -> None:
    """Run the many-jobs scenario (8 jobs, 8 slots or 16 with 2 CPUs)."""
    print("\n" + "=" * 70)
    market = create_many_jobs_example(num_cpus=2 if two_cpus else 1)
    slots_desc = f"{len(market.slots)} (2 CPUs × 8 times)" if two_cpus else f"{len(market.slots)} (9am-5pm)"
    print("\nMany-Jobs Example:")
    print(f"  Slots: {slots_desc}, Reserve: $3.00/hour")
    print(f"  Agents: {len(market.agents)} jobs")
    for agent in market.agents:
        print(f"    {agent.name}: {agent.required_slots}h, deadline slot_{agent.deadline_slot_id}, w=${agent.worth:.2f}")
    print(f"\nRunning Ascending Auction with eps=${epsilon:.2f}...")
    auction = AscendingAuction(epsilon=epsilon)
    result = auction.run(market)
    if show_trace:
        result.print_trace(max_rounds=150)
    print("\nFinal market state:")
    print_final_market_state(result)
    metrics = compute_metrics(result, epsilon)
    print_metrics_report(metrics)
    return result, metrics


def run_duplicate_example_1(epsilon: float = 0.25, show_trace: bool = True, two_cpus: bool = False) -> None:
    """Run duplicate of book example 1 (8 jobs: Job1–Job4 each duplicated as Job5–Job8)."""
    print("\n" + "=" * 70)
    market = create_duplicate_example_1(num_cpus=2 if two_cpus else 1)
    slots_desc = f"{len(market.slots)} (2 CPUs x 8 times)" if two_cpus else f"{len(market.slots)} (9am-5pm)"
    print("\nDuplicate Example 1 (same job types as book example 1, each doubled):")
    print(f"  Slots: {slots_desc}, Reserve: $3.00/hour")
    print(f"  Agents: {len(market.agents)} jobs (Job1=Job5, Job2=Job6, Job3=Job7, Job4=Job8)")
    for agent in market.agents:
        print(f"    {agent.name}: {agent.required_slots}h, deadline slot_{agent.deadline_slot_id}, w=${agent.worth:.2f}")
    print(f"\nRunning Ascending Auction with eps=${epsilon:.2f}...")
    auction = AscendingAuction(epsilon=epsilon)
    result = auction.run(market)
    if show_trace:
        result.print_trace(max_rounds=150)
    print("\nFinal market state:")
    print_final_market_state(result)
    metrics = compute_metrics(result, epsilon)
    print_metrics_report(metrics)
    return result, metrics


def run_book_example_2(epsilon: float = 0.25, two_cpus: bool = False) -> AuctionResult | None:
    """
    Run the Table 2.1 example (2 slots, 2 jobs).
    With two_cpus=True: 4 slots (2 times × 2 CPUs), same 2 jobs.
    """
    print("\n" + "=" * 70)

    market = create_book_example_2_two_cpus() if two_cpus else create_book_example_2()
    slots_desc = "4 (2 CPUs × 9am, 10am)" if two_cpus else "2 (9am, 10am)"
    print(f"\nMarket Configuration:")
    print(f"  Slots: {slots_desc}, Reserve: $3.00/hour")
    print(f"  Agents:")
    for agent in market.agents:
        print(f"    {agent.name}: λ={agent.required_slots}, d=slot_{agent.deadline_slot_id}, w=${agent.worth:.2f}")

    print(f"\nRunning Ascending Auction with ε={epsilon:.2f}...")
    auction = AscendingAuction(epsilon=epsilon, max_iterations=100)
    result = auction.run(market)
    print(f"  Iterations: {result.iterations}")
    print(f"  Solution: ${result.final_solution_value:.2f}")
    print_final_market_state(result)
    return result


def run_book_example_3() -> AuctionResult:
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

    print(f"\nRunning Ascending Auction with ε=$0.25...")
    auction = AscendingAuction(epsilon=0.25)
    result = auction.run(market)

    print(f"\nAuction Result:")
    print(f"  Solution: ${result.final_solution_value:.2f}")
    print("  Final market state:")
    print_final_market_state(result)
    print("\n  Note: By adjusting values, auction outcome can be made arbitrarily far from optimal.")
    return result


def run_competitive_scenario(epsilon: float = 0.25) -> AuctionResult:
    """Run the competitive scenario (high competition for early slots) with given epsilon."""
    print("\n" + "=" * 70)
    market = create_competitive_scenario()
    print(f"\nMarket Configuration (Competitive): 4 slots, 4 agents")
    for agent in market.agents:
        print(f"  {agent.name}: λ={agent.required_slots}, d=slot_{agent.deadline_slot_id}, w=${agent.worth:.2f}")
    print(f"\nRunning Ascending Auction with ε={epsilon:.2f}...")
    auction = AscendingAuction(epsilon=epsilon, max_iterations=100)
    result = auction.run(market)
    print(f"  Iterations: {result.iterations}")
    print(f"  Solution: ${result.final_solution_value:.2f}")
    print_final_market_state(result)
    return result


def run_24h_night_discount_scenario(
    epsilon: float = 0.25, two_cpus: bool = False
) -> AuctionResult:
    """Run the 24-hour scenario with nighttime-discounted reserves and 20 jobs."""
    print("\n" + "=" * 70)
    num_cpus = 2 if two_cpus else 1
    market = create_24h_night_discount_scenario(num_agents=20, num_cpus=num_cpus)
    slots_desc = "48 (24 × 2 CPUs)" if two_cpus else "24 (midnight–11 PM)"
    print(f"\nMarket Configuration (24h, night discount): {slots_desc} slots, 20 jobs")
    print("  Night slots (10 PM–6 AM): discounted reserve. Day slots: full reserve.")
    print(f"\nRunning Ascending Auction with ε={epsilon:.2f}...")
    auction = AscendingAuction(epsilon=epsilon, max_iterations=200)
    result = auction.run(market)
    print(f"  Iterations: {result.iterations}")
    print(f"  Solution: ${result.final_solution_value:.2f}")
    print_final_market_state(result)
    return result


# Epsilon values used in sensitivity analysis [0.05, 0.1, 0.15, ..., 2.95, 3.00].
SENSITIVITY_EPSILONS = [round(0.05 + i * 0.05, 2) for i in range(60)]

# Experiments to include in sensitivity analysis: (label, market_factory)
# Single-CPU and multi-CPU (2 CPUs) variants where applicable.
SENSITIVITY_EXPERIMENTS = [
    ("Example 1", create_book_example_1),
    ("Example 1 (2 CPUs)", create_book_example_1_two_cpus),
    ("Example 2", create_book_example_2),
    ("Example 2 (2 CPUs)", create_book_example_2_two_cpus),
    ("Example 3", create_book_example_3),
    ("Example 4 (many jobs)", create_many_jobs_example),
    ("Example 4 (many jobs, 2 CPUs)", lambda: create_many_jobs_example(num_cpus=2)),
    ("Example 5 (duplicate ex1)", lambda: create_duplicate_example_1(num_cpus=1)),
    ("Example 5 (duplicate ex1, 2 CPUs)", lambda: create_duplicate_example_1(num_cpus=2)),
    ("Competitive", create_competitive_scenario),
    ("Scalability (24h, 20 jobs)", lambda: create_24h_night_discount_scenario(num_agents=20)),
    ("Scalability (24h, 20 jobs, 2 CPUs)", lambda: create_24h_night_discount_scenario(num_agents=20, num_cpus=2)),
]


def get_sensitivity_experiment(example: int, two_cpus: bool) -> tuple[str, Callable[[], Market]]:
    """Return (label, market_factory) for sensitivity analysis for the given example and CPU count."""
    if example == 1:
        return ("Example 1 (2 CPUs)", create_book_example_1_two_cpus) if two_cpus else ("Example 1", create_book_example_1)
    if example == 2:
        return ("Example 2 (2 CPUs)", create_book_example_2_two_cpus) if two_cpus else ("Example 2", create_book_example_2)
    if example == 3:
        return ("Example 3", create_book_example_3)  # no 2-CPU variant
    if example == 4:
        return ("Example 4 (many jobs, 2 CPUs)", lambda: create_many_jobs_example(num_cpus=2)) if two_cpus else ("Example 4 (many jobs)", create_many_jobs_example)
    if example == 5:
        return ("Example 5 (duplicate ex1, 2 CPUs)", lambda: create_duplicate_example_1(num_cpus=2)) if two_cpus else ("Example 5 (duplicate ex1)", lambda: create_duplicate_example_1(num_cpus=1))
    if example == 6:
        return ("Competitive", create_competitive_scenario)
    if example == 7:
        return ("Scalability (24h, 20 jobs, 2 CPUs)", lambda: create_24h_night_discount_scenario(num_agents=20, num_cpus=2)) if two_cpus else ("Scalability (24h, 20 jobs)", lambda: create_24h_night_discount_scenario(num_agents=20))
    raise ValueError(f"Unknown example: {example}")


def run_epsilon_sensitivity_single(
    name: str,
    market_fn: Callable[[], Market],
    show_plots: bool = True,
    save_plots: bool = False,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Run epsilon sensitivity analysis for a single experiment.
    """
    print("\n" + "=" * 70)
    print(f"EPSILON SENSITIVITY ANALYSIS — {name}")
    print("=" * 70)

    epsilons = SENSITIVITY_EPSILONS
    print(f"\nEpsilon values: {epsilons}")
    print("Running sensitivity...\n")

    market = market_fn()
    sensitivity = run_epsilon_sensitivity(market, epsilons)

    print(f"  {name}:")
    print(f"    {'Eps':<8} {'Iters':<10} {'Solution ($)':<12}")
    for i, eps in enumerate(sensitivity.epsilons):
        print(f"    {eps:<8.2f} {sensitivity.iterations[i]:<10} {sensitivity.solution_values[i]:<12.2f}")
    print()

    if show_plots or save_plots:
        fig = plot_epsilon_sensitivity(sensitivity, title=f"Epsilon Sensitivity — {name}", save_path=None)
        if fig:
            if save_plots and output_dir:
                safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                path = output_dir / f"epsilon_sensitivity_{safe_name}.png"
                fig.savefig(str(path), dpi=150, bbox_inches="tight")
                print(f"Sensitivity plot saved to {path}")
            if show_plots:
                plt.show()


def run_epsilon_sensitivity_analysis(
    show_plots: bool = True,
    save_plots: bool = False,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Run epsilon sensitivity analysis for all experiments (examples 1–5,
    including 2-CPU variants for 1, 2, 4, 5).
    """
    print("\n" + "=" * 70)
    print("EPSILON SENSITIVITY ANALYSIS (ALL EXPERIMENTS)")
    print("=" * 70)

    epsilons = SENSITIVITY_EPSILONS
    print(f"\nEpsilon values: {epsilons}")
    print("Running sensitivity for each experiment...\n")

    results: list[tuple[str, EpsilonSensitivityResult]] = []
    for name, market_fn in SENSITIVITY_EXPERIMENTS:
        market = market_fn()
        sensitivity = run_epsilon_sensitivity(market, epsilons)
        results.append((name, sensitivity))

        print(f"  {name}:")
        print(f"    {'Eps':<8} {'Iters':<10} {'Solution ($)':<12}")
        for i, eps in enumerate(sensitivity.epsilons):
            print(f"    {eps:<8.2f} {sensitivity.iterations[i]:<10} {sensitivity.solution_values[i]:<12.2f}")
        print()

    if show_plots or save_plots:
        fig = plot_epsilon_sensitivity_all(results, save_path=None)
        if fig:
            if save_plots and output_dir:
                path = output_dir / "epsilon_sensitivity_all.png"
                fig.savefig(str(path), dpi=150, bbox_inches="tight")
                print(f"Sensitivity plot saved to {path}")
            if show_plots:
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
        plot_allocation_and_prices(result1, title="Allocation and Prices - Example 1 (ε=0.25)",
                                  save_path=str(output_dir / "ex1_allocation_025.png"))
    # Example 1 with ε=1.0
    print("\n" + "#" * 70)
    print("# Running Book Example 1 with ε=1.0")
    print("#" * 70)
    result1b, metrics1b = run_book_example_1(epsilon=1.0, show_trace=True)
    if save_plots:
        plot_price_evolution(result1b, title="Price Evolution - Example 1 (ε=1.0)",
                            save_path=str(output_dir / "ex1_prices_10.png"))
        plot_allocation_and_prices(result1b, title="Allocation and Prices - Example 1 (ε=1.0)",
                                  save_path=str(output_dir / "ex1_allocation_10.png"))
    # Example 2
    print("\n" + "#" * 70)
    print("# Running Book Example 2 with ε=0.25")
    print("#" * 70)
    result2 = run_book_example_2(epsilon=0.25)
    if save_plots and result2 is not None:
        plot_allocation_and_prices(result2, title="Allocation and Prices - Example 2 (ε=0.25)",
                                  save_path=str(output_dir / "ex2_allocation_eps0.25.png"))
    # Example 3 (suboptimal)
    print("\n" + "#" * 70)
    print("# Running Book Example 3 (Suboptimal Case)")
    print("#" * 70)
    result3 = run_book_example_3()
    if save_plots and result3 is not None:
        plot_allocation_and_prices(result3, title="Allocation and Prices - Example 3",
                                  save_path=str(output_dir / "ex3_allocation.png"))
    
    # Epsilon sensitivity (all experiments)
    print("\n" + "#" * 70)
    print("# Running Epsilon Sensitivity Analysis (All Experiments)")
    print("#" * 70)
    run_epsilon_sensitivity_analysis(
        show_plots=not save_plots,
        save_plots=save_plots,
        output_dir=output_dir if save_plots else None,
    )
    if save_plots:
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
  python main.py --example 1 --two-cpus  Run example 1 with 2 identical CPUs
  python main.py --example 4            Run many-jobs example (8 jobs)
  python main.py --example 4 --two-cpus Run many jobs with 2 CPUs
  python main.py --example 5            Run duplicate of example 1 (8 identical job types)
  python main.py --example 5 --two-cpus Duplicate ex1 with 2 CPUs
  python main.py --example 2 --two-cpus Example 2 with 2 CPUs (4 slots)
  python main.py --sensitivity       Run epsilon sensitivity (all experiments)
  python main.py -s -e 1             Sensitivity for Example 1 only
  python main.py -s -e 4 --two-cpus Sensitivity for Example 4 with 2 CPUs
  python main.py --all --save        Run all experiments and save plots
        """
    )
    
    parser.add_argument("--example", "-e", type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                       help="Run specific example (1-3: book, 4: many jobs, 5: duplicate ex1, 6: competitive, 7: 24h night discount)")
    parser.add_argument("--eps", type=float, default=0.25,
                       help="Epsilon value for auction (default: 0.25)")
    parser.add_argument("--sensitivity", "-s", action="store_true",
                       help="Run epsilon sensitivity (all experiments, or one if --example is set)")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Run all experiments")
    parser.add_argument("--save", action="store_true",
                       help="Save plots to output directory")
    parser.add_argument("--no-plots", action="store_true",
                       help="Disable plot display")
    parser.add_argument("--two-cpus", action="store_true",
                       help="Use 2 identical CPUs (examples 1, 2, 4, 5)")
    
    args = parser.parse_args()
    
    # Default behavior: run all book examples
    if not any([args.example, args.sensitivity, args.all]):
        args.all = True
    
    try:
        if args.all:
            run_all_experiments(save_plots=args.save)
        elif args.sensitivity and args.example is not None:
            name, market_fn = get_sensitivity_experiment(args.example, args.two_cpus)
            out_dir = Path("output") if args.save else None
            if args.save:
                out_dir.mkdir(exist_ok=True)
            run_epsilon_sensitivity_single(
                name=name,
                market_fn=market_fn,
                show_plots=not args.no_plots,
                save_plots=args.save,
                output_dir=out_dir,
            )
        elif args.sensitivity:
            out_dir = Path("output") if args.save else None
            if args.save:
                out_dir.mkdir(exist_ok=True)
            run_epsilon_sensitivity_analysis(
                show_plots=not args.no_plots,
                save_plots=args.save,
                output_dir=out_dir,
            )
        elif args.example == 1:
            result1, _ = run_book_example_1(epsilon=args.eps, two_cpus=args.two_cpus)
            if args.save:
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                suffix = "_2cpus" if args.two_cpus else ""
                plot_allocation_and_prices(result1, title=f"Allocation and Prices - Example 1 (ε={args.eps})",
                                          save_path=str(output_dir / f"ex1_allocation_eps{args.eps}{suffix}.png"))
            if not args.no_plots:
                plt.show()
        elif args.example == 2:
            result2 = run_book_example_2(epsilon=args.eps, two_cpus=args.two_cpus)
            if args.save and result2 is not None:
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                suffix = "_2cpus" if args.two_cpus else ""
                plot_allocation_and_prices(result2, title=f"Allocation and Prices - Example 2 (ε={args.eps}){suffix}",
                                          save_path=str(output_dir / f"ex2_allocation_eps{args.eps}{suffix}.png"))
                print(f"\nPlot saved to {output_dir / f'ex2_allocation_eps{args.eps}{suffix}.png'}")
        elif args.example == 3:
            result3 = run_book_example_3()
            if args.save:
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                plot_allocation_and_prices(result3, title="Allocation and Prices - Example 3",
                                          save_path=str(output_dir / "ex3_allocation.png"))
                print(f"\nPlot saved to {output_dir / 'ex3_allocation.png'}")
        elif args.example == 4:
            result4, _ = run_many_jobs_example(epsilon=args.eps, two_cpus=args.two_cpus)
            if args.save:
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                suffix = "_2cpus" if args.two_cpus else ""
                plot_allocation_and_prices(result4, title=f"Many Jobs (ε={args.eps}){suffix}",
                                          save_path=str(output_dir / f"ex4_many_jobs{suffix}.png"))
            if not args.no_plots:
                plt.show()
        elif args.example == 5:
            result5, _ = run_duplicate_example_1(epsilon=args.eps, two_cpus=args.two_cpus)
            if args.save:
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                suffix = "_2cpus" if args.two_cpus else ""
                plot_allocation_and_prices(result5, title=f"Duplicate Ex1 (ε={args.eps}){suffix}",
                                          save_path=str(output_dir / f"ex5_duplicate_ex1_eps{args.eps}{suffix}.png"))
            if not args.no_plots:
                plt.show()
        elif args.example == 6:
            result6 = run_competitive_scenario(epsilon=args.eps)
            if args.save and result6 is not None:
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                plot_allocation_and_prices(result6, title=f"Allocation and Prices - Competitive (ε={args.eps})",
                                          save_path=str(output_dir / f"competitive_allocation_eps{args.eps}.png"))
                print(f"\nPlot saved to {output_dir / f'competitive_allocation_eps{args.eps}.png'}")
            if not args.no_plots:
                plt.show()
        elif args.example == 7:
            result7 = run_24h_night_discount_scenario(epsilon=args.eps, two_cpus=args.two_cpus)
            if args.save and result7 is not None:
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                suffix = "_2cpus" if args.two_cpus else ""
                plot_allocation_and_prices(result7, title=f"Allocation and Prices - 24h Night Discount (ε={args.eps}){suffix}",
                                          save_path=str(output_dir / f"scalability_24h_allocation_eps{args.eps}{suffix}.png"))
                print(f"\nPlot saved to {output_dir / f'scalability_24h_allocation_eps{args.eps}{suffix}.png'}")
            if not args.no_plots:
                plt.show()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
