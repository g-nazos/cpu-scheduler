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
        ax.plot(rounds, prices, label=f"Slot {i} ({slot.time_label})", 
                color=colors[i], linewidth=2)
    
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Bid Price ($)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_allocation_timeline(
    market: Market,
    title: str = "Allocation Timeline",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a Gantt-chart style allocation timeline.
    
    Args:
        market: Market with allocations
        title: Plot title
        save_path: Path to save figure (if provided)
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(market.agents) + 1))
    agent_colors = {agent.agent_id: colors[i] for i, agent in enumerate(market.agents)}
    agent_colors[None] = colors[-1]  # Unallocated
    
    # Plot each slot
    for slot in market.slots:
        owner = market.get_slot_owner(slot)
        owner_id = owner.agent_id if owner else None
        color = agent_colors[owner_id]
        
        ax.barh(slot.slot_id, 1, left=0, height=0.8, 
                color=color, edgecolor="black", linewidth=1)
        
        # Add label
        label = owner.name if owner else "Unalloc"
        ax.text(0.5, slot.slot_id, label, ha="center", va="center", fontsize=10)
    
    # Y-axis labels (slot times)
    ax.set_yticks([s.slot_id for s in market.slots])
    ax.set_yticklabels([s.time_label for s in market.slots])
    
    ax.set_xlabel("Allocation", fontsize=12)
    ax.set_ylabel("Time Slot", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Legend
    patches = [mpatches.Patch(color=agent_colors[a.agent_id], label=a.name) 
               for a in market.agents]
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
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.Set3(np.linspace(0, 1, len(market.agents) + 1))
    agent_colors = {agent.agent_id: colors[i] for i, agent in enumerate(market.agents)}
    agent_colors[None] = colors[-1]

    for slot in market.slots:
        owner = market.get_slot_owner(slot)
        owner_id = owner.agent_id if owner else None
        color = agent_colors[owner_id]
        price = market.bid_prices.get(slot.slot_id, slot.reserve_price)
        ax.barh(slot.slot_id, 1, left=0, height=0.8,
                color=color, edgecolor="black", linewidth=1)
        label = (owner.name if owner else "Unalloc") + f"  ${price:.2f}"
        ax.text(0.5, slot.slot_id, label, ha="center", va="center", fontsize=10)

    ax.set_yticks([s.slot_id for s in market.slots])
    ax.set_yticklabels([s.time_label for s in market.slots])
    ax.set_ylabel("Time Slot", fontsize=12)
    ax.set_xlabel("CPU", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    patches = [mpatches.Patch(color=agent_colors[a.agent_id], label=a.name)
               for a in market.agents]
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
    Plot results of epsilon sensitivity analysis.

    Args:
        sensitivity: Results from epsilon sensitivity analysis
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.plot(sensitivity.epsilons, sensitivity.iterations, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Epsilon (ε)", fontsize=11)
    ax1.set_ylabel("Iterations", fontsize=11)
    ax1.set_title("Iterations vs Epsilon", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    colors = ["green" if eq else "red" for eq in sensitivity.equilibrium_achieved]
    ax2.bar(range(len(sensitivity.epsilons)),
            [1 if eq else 0 for eq in sensitivity.equilibrium_achieved],
            color=colors, edgecolor="black")
    ax2.set_xticks(range(len(sensitivity.epsilons)))
    ax2.set_xticklabels([f"{e:.2f}" for e in sensitivity.epsilons], rotation=45)
    ax2.set_xlabel("Epsilon (ε)", fontsize=11)
    ax2.set_ylabel("Equilibrium Achieved", fontsize=11)
    ax2.set_title("Equilibrium Achievement", fontsize=12)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["No", "Yes"])

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

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
    ax1.set_ylabel("Slot ID", fontsize=11)
    ax1.set_title("Allocation Over Rounds", fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label("Agent ID (0 = unallocated)")
    
    # Bottom: Prices over time
    ax2 = axes[1]
    for i in range(num_slots):
        prices = [r.bid_prices[i] for r in rounds_to_show]
        ax2.plot(range(len(rounds_to_show)), prices, 
                 label=f"Slot {i}", linewidth=1.5)
    
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
