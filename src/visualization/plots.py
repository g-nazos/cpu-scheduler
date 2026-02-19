"""
Visualization utilities for auction results.

Creates plots for price evolution, allocation timelines,
solution value comparison, and convergence analysis.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.models.market import Market
from src.auction.ascending import AuctionResult
from src.experiments.metrics import EpsilonSensitivityResult


def plot_price_evolution(
    result: AuctionResult,
    title: str = "Price Evolution During Auction",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the evolution of bid prices over auction rounds.
    
    Args:
        result: Auction result with round history
        title: Plot title
        save_path: Path to save figure (if provided)
        
    Returns:
        matplotlib Figure
    """
    if not result.rounds:
        print("No rounds to plot")
        return None
    
    # Extract price history
    num_slots = len(result.rounds[0].bid_prices)
    rounds = range(len(result.rounds))
    
    prices_by_slot = [[] for _ in range(num_slots)]
    for r in result.rounds:
        for i, price in enumerate(r.bid_prices):
            prices_by_slot[i].append(price)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_slots))
    for i, prices in enumerate(prices_by_slot):
        slot = result.market.slots[i]
        ax.plot(rounds, prices, label=_slot_ylabel(slot), color=colors[i], linewidth=2)
    
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Bid Price ($)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def _slot_ylabel(slot) -> str:
    """Label for y-axis: show CPU when multi-CPU."""
    if getattr(slot, "time_index", None) is not None:
        return f"CPU{slot.cpu_id} {slot.time_label}"
    return slot.time_label


def _is_multi_cpu(market: Market) -> bool:
    """True if market has multiple CPUs (slots have time_index set)."""
    if not market.slots:
        return False
    return getattr(market.slots[0], "time_index", None) is not None


def _slots_by_cpu(market: Market) -> dict[int, list]:
    """Group slots by cpu_id, sorted by time_index (for multi-CPU). Returns {cpu_id: [slot, ...]}."""
    by_cpu: dict[int, list] = {}
    for s in market.slots:
        c = getattr(s, "cpu_id", 0)
        by_cpu.setdefault(c, []).append(s)
    for c in by_cpu:
        by_cpu[c] = sorted(by_cpu[c], key=lambda s: s.get_time_index() if hasattr(s, "get_time_index") else s.slot_id)
    return by_cpu


def plot_allocation_timeline(
    market: Market,
    title: str = "Allocation Timeline",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a Gantt-chart style allocation timeline.
    For 2 CPUs: two panels side by side (CPU 0 left, CPU 1 right), same time on y-axis.
    """
    colors = plt.cm.Set3(np.linspace(0, 1, len(market.agents) + 1))
    agent_colors = {agent.agent_id: colors[i] for i, agent in enumerate(market.agents)}
    agent_colors[None] = colors[-1]

    if _is_multi_cpu(market):
        by_cpu = _slots_by_cpu(market)
        cpus = sorted(by_cpu.keys())
        n_cpus = len(cpus)
        n_times = max(len(by_cpu[c]) for c in cpus) if cpus else 0
        fig, axes = plt.subplots(1, n_cpus, figsize=(5 * n_cpus, max(5, n_times * 0.6)), sharey=True)
        if n_cpus == 1:
            axes = [axes]
        for idx, cpu in enumerate(cpus):
            ax = axes[idx]
            slots = by_cpu[cpu]
            for slot in slots:
                owner = market.get_slot_owner(slot)
                owner_id = owner.agent_id if owner else None
                color = agent_colors[owner_id]
                y = slot.get_time_index()
                ax.barh(y, 1, left=0, height=0.8, color=color, edgecolor="black", linewidth=1)
                label = owner.name if owner else "Unalloc"
                ax.text(0.5, y, label, ha="center", va="center", fontsize=10)
            ax.set_yticks([s.get_time_index() for s in slots])
            ax.set_yticklabels([s.time_label for s in slots])
            ax.set_ylabel("Time" if idx == 0 else "")
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.set_title(f"CPU {cpu}", fontsize=12)
        fig.suptitle(title, fontsize=14)
        patches = [mpatches.Patch(color=agent_colors[a.agent_id], label=a.name) for a in market.agents]
        patches.append(mpatches.Patch(color=agent_colors[None], label="Unallocated"))
        axes[-1].legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1))
    else:
        fig, ax = plt.subplots(figsize=(12, max(6, len(market.slots) * 0.35)))
        for slot in market.slots:
            owner = market.get_slot_owner(slot)
            owner_id = owner.agent_id if owner else None
            color = agent_colors[owner_id]
            ax.barh(slot.slot_id, 1, left=0, height=0.8, color=color, edgecolor="black", linewidth=1)
            label = owner.name if owner else "Unalloc"
            ax.text(0.5, slot.slot_id, label, ha="center", va="center", fontsize=10)
        ax.set_yticks([s.slot_id for s in market.slots])
        ax.set_yticklabels([s.time_label for s in market.slots])
        ax.set_xlabel("Allocation", fontsize=12)
        ax.set_ylabel("Time Slot", fontsize=12)
        ax.set_title(title, fontsize=14)
        patches = [mpatches.Patch(color=agent_colors[a.agent_id], label=a.name) for a in market.agents]
        patches.append(mpatches.Patch(color=agent_colors[None], label="Unallocated"))
        ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.set_xlim(0, 1)
        ax.set_xticks([])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_allocation_and_prices(
    result: AuctionResult,
    title: str = "Allocation and Slot Prices",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot allocation with bid price shown inside each slot bar.

    Single Gantt-style panel: each slot bar shows owner and price (e.g. "Job2  $5.00").

    Args:
        result: Auction result (market with allocations and bid_prices)
        title: Plot title
        save_path: Path to save figure (if provided)

    Returns:
        matplotlib Figure
    """
    market = result.market
    colors = plt.cm.Set3(np.linspace(0, 1, len(market.agents) + 1))
    agent_colors = {agent.agent_id: colors[i] for i, agent in enumerate(market.agents)}
    agent_colors[None] = colors[-1]

    if _is_multi_cpu(market):
        by_cpu = _slots_by_cpu(market)
        cpus = sorted(by_cpu.keys())
        n_cpus = len(cpus)
        n_times = max(len(by_cpu[c]) for c in cpus) if cpus else 0
        fig, axes = plt.subplots(1, n_cpus, figsize=(5 * n_cpus, max(5, n_times * 0.6)), sharey=True)
        if n_cpus == 1:
            axes = [axes]
        for idx, cpu in enumerate(cpus):
            ax = axes[idx]
            slots = by_cpu[cpu]
            for slot in slots:
                owner = market.get_slot_owner(slot)
                owner_id = owner.agent_id if owner else None
                color = agent_colors[owner_id]
                price = market.bid_prices.get(slot.slot_id, slot.reserve_price)
                y = slot.get_time_index()
                ax.barh(y, 1, left=0, height=0.8, color=color, edgecolor="black", linewidth=1)
                label = (owner.name if owner else "Unalloc") + f"  ${price:.2f}"
                ax.text(0.5, y, label, ha="center", va="center", fontsize=10)
            ax.set_yticks([s.get_time_index() for s in slots])
            ax.set_yticklabels([s.time_label for s in slots])
            ax.set_ylabel("Time" if idx == 0 else "")
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.set_title(f"CPU {cpu}", fontsize=12)
        fig.suptitle(title, fontsize=14)
        patches = [mpatches.Patch(color=agent_colors[a.agent_id], label=a.name) for a in market.agents]
        patches.append(mpatches.Patch(color=agent_colors[None], label="Unallocated"))
        axes[-1].legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1))
    else:
        fig, ax = plt.subplots(figsize=(12, max(6, len(market.slots) * 0.35)))
        for slot in market.slots:
            owner = market.get_slot_owner(slot)
            owner_id = owner.agent_id if owner else None
            color = agent_colors[owner_id]
            price = market.bid_prices.get(slot.slot_id, slot.reserve_price)
            ax.barh(slot.slot_id, 1, left=0, height=0.8, color=color, edgecolor="black", linewidth=1)
            label = (owner.name if owner else "Unalloc") + f"  ${price:.2f}"
            ax.text(0.5, slot.slot_id, label, ha="center", va="center", fontsize=10)
        ax.set_yticks([s.slot_id for s in market.slots])
        ax.set_yticklabels([s.time_label for s in market.slots])
        ax.set_ylabel("Time Slot", fontsize=12)
        ax.set_xlabel("CPU", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        patches = [mpatches.Patch(color=agent_colors[a.agent_id], label=a.name) for a in market.agents]
        patches.append(mpatches.Patch(color=agent_colors[None], label="Unallocated"))
        ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_solution_value_comparison(
    auction_solution_value: float,
    reserve_solution_value: float,
    title: str = "Solution Value Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bar chart comparing auction solution value to reserve-only.

    Args:
        auction_solution_value: Solution value from auction
        reserve_solution_value: Solution value if all slots unallocated
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["Reserve Only", "Auction"]
    values = [reserve_solution_value, auction_solution_value]
    colors = ["#ff9999", "#66b3ff"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"${value:.2f}", ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("Solution Value ($)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, max(values) * 1.15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_epsilon_sensitivity(
    sensitivity: EpsilonSensitivityResult,
    title: str = "Epsilon Sensitivity Analysis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot epsilon sensitivity: epsilon vs iterations and epsilon vs solution value.

    Args:
        sensitivity: Results from epsilon sensitivity analysis
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.plot(sensitivity.epsilons, sensitivity.iterations, "bo-", linewidth=1, markersize=4)
    ax1.set_xlabel("Epsilon (ε)", fontsize=11)
    ax1.set_ylabel("Iterations", fontsize=11)
    ax1.set_title("Epsilon vs Iterations", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(sensitivity.epsilons, sensitivity.solution_values, "go-", linewidth=1, markersize=4)
    ax2.set_xlabel("Epsilon (ε)", fontsize=11)
    ax2.set_ylabel("Solution Value ($)", fontsize=11)
    ax2.set_title("Epsilon vs Solution Value", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.suptitle(title, fontsize=14, y=0.98)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_epsilon_sensitivity_all(
    results: list[tuple[str, EpsilonSensitivityResult]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot epsilon sensitivity for multiple experiments. Each row: epsilon vs iterations,
    epsilon vs solution value.
    """
    n = len(results)
    if n == 0:
        return None
    row_h = 3.0 if n > 5 else 4.0
    fig, axes = plt.subplots(n, 2, figsize=(12, row_h * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, (name, sensitivity) in enumerate(results):
        ax1, ax2 = axes[i, 0], axes[i, 1]
        ax1.plot(sensitivity.epsilons, sensitivity.iterations, "bo-", linewidth=2, markersize=6)
        ax1.set_ylabel("Iterations", fontsize=10)
        ax1.set_title(f"{name}: ε vs Iterations", fontsize=11)
        ax1.grid(True, alpha=0.3)
        if i == 0:
            ax1.set_xlabel("Epsilon (ε)", fontsize=10)
        ax2.plot(sensitivity.epsilons, sensitivity.solution_values, "go-", linewidth=2, markersize=6)
        ax2.set_ylabel("Solution Value ($)", fontsize=10)
        ax2.set_title(f"{name}: ε vs Solution Value", fontsize=11)
        ax2.grid(True, alpha=0.3)
        if i == 0:
            ax2.set_xlabel("Epsilon (ε)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("Epsilon Sensitivity — All Experiments", fontsize=14, y=0.98)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_convergence_trace(
    result: AuctionResult,
    max_rounds: int = 50,
    title: str = "Auction Convergence Trace",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a detailed trace of auction convergence.
    
    Args:
        result: Auction result
        max_rounds: Maximum rounds to show
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    if not result.rounds:
        print("No rounds to plot")
        return None
    
    rounds_to_show = result.rounds[:max_rounds]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top: Allocations over time
    ax1 = axes[0]
    num_agents = len(result.market.agents)
    num_slots = len(result.market.slots)
    
    # Create allocation matrix (rounds x slots) -> agent_id
    alloc_matrix = np.zeros((len(rounds_to_show), num_slots))
    for r_idx, r in enumerate(rounds_to_show):
        for agent_id, slot_ids in r.allocations.items():
            for slot_id in slot_ids:
                alloc_matrix[r_idx, slot_id] = agent_id
    
    im = ax1.imshow(alloc_matrix.T, aspect="auto", cmap="tab10",
                    vmin=0, vmax=num_agents + 1)
    ax1.set_ylabel("Slot", fontsize=11)
    ax1.set_title("Allocation Over Rounds", fontsize=12)
    slots = result.market.slots
    if num_slots <= 16:
        ax1.set_yticks(range(num_slots))
        ax1.set_yticklabels([_slot_ylabel(slots[i]) for i in range(num_slots)], fontsize=8)
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label("Agent ID (0 = unallocated)")

    ax2 = axes[1]
    for i in range(num_slots):
        prices = [r.bid_prices[i] for r in rounds_to_show]
        ax2.plot(range(len(rounds_to_show)), prices,
                 label=_slot_ylabel(slots[i]) if i < len(slots) else f"Slot {i}", linewidth=1.5)
    
    ax2.set_xlabel("Round", fontsize=11)
    ax2.set_ylabel("Bid Price ($)", fontsize=11)
    ax2.set_title("Price Evolution", fontsize=12)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig
