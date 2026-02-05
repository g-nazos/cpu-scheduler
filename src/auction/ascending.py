"""
Ascending Auction Algorithm for the Scheduling Problem.

Implementation of Figure 2.7 from Section 2.3.3 of "Multiagent Systems"
by Shoham & Leyton-Brown.
"""

import logging
from dataclasses import dataclass, field
from typing import FrozenSet, Optional

from src.models.agent import Agent
from src.models.slot import Slot
from src.models.market import Market


logger = logging.getLogger(__name__)


@dataclass
class AuctionRound:
    """Record of a single auction round for tracing."""
    round_num: int
    bidder: Agent
    slots_bid_on: FrozenSet[Slot]
    allocations: dict[int, list[int]]  # agent_id -> list of slot_ids
    bid_prices: list[float]


@dataclass
class AuctionResult:
    """Result of running the ascending auction."""
    market: Market
    rounds: list[AuctionRound]
    converged: bool
    iterations: int
    final_welfare: float
    
    def print_trace(self, max_rounds: Optional[int] = None) -> None:
        """Print a trace of the auction similar to the book's tables."""
        print("\nAuction Trace:")
        print("-" * 80)
        header = f"{'Round':<6} {'Bidder':<10} {'Slots Bid On':<15} {'Allocations':<30} {'Prices'}"
        print(header)
        print("-" * 80)
        
        rounds_to_print = self.rounds[:max_rounds] if max_rounds else self.rounds
        for r in rounds_to_print:
            slot_ids = sorted([s.slot_id for s in r.slots_bid_on])
            alloc_str = str({k: v for k, v in r.allocations.items() if v})
            prices_str = [f"{p:.2f}" for p in r.bid_prices]
            print(f"{r.round_num:<6} {r.bidder.name:<10} {str(slot_ids):<15} {alloc_str:<30} {prices_str}")
        
        if max_rounds and len(self.rounds) > max_rounds:
            print(f"... ({len(self.rounds) - max_rounds} more rounds)")
        
        print("-" * 80)
        print(f"Converged: {self.converged}, Iterations: {self.iterations}, "
              f"Final Welfare: {self.final_welfare:.2f}")


class AscendingAuction:
    """
    Ascending Auction Algorithm from Figure 2.7.
    
    The algorithm:
    1. Initialize bid prices to reserve prices, allocations to empty
    2. Repeat until no change:
       a. For each agent i:
          - Compute ask prices (current bid for held slots, bid + epsilon for others)
          - Find best bundle S* maximizing surplus
          - For new slots in S* minus F_i: increment bid, remove from other agents
          - Update F_i = S*
    3. Return final allocations and prices
    """
    
    def __init__(self, epsilon: float = 0.25, max_iterations: int = 10000):
        """
        Initialize the auction.
        
        Args:
            epsilon: Price increment (ε)
            max_iterations: Maximum iterations to prevent infinite loops
        """
        self.epsilon = epsilon
        self.max_iterations = max_iterations
    
    def run(self, market: Market, verbose: bool = False) -> AuctionResult:
        """
        Run the ascending auction algorithm.
        
        Args:
            market: Initial market state
            verbose: Whether to print progress
            
        Returns:
            AuctionResult with final state and trace
        """
        # Work on a copy to not modify the original
        market = market.copy()
        
        # Initialize: bid prices = reserve prices (done in Market.__post_init__)
        # Initialize: all allocations empty (done in Market.__post_init__)
        
        rounds: list[AuctionRound] = []
        round_num = 0
        iteration = 0
        
        while iteration < self.max_iterations:
            # Track if any change occurred in this full pass
            any_change = False
            
            for agent in market.agents:
                round_num += 1
                
                # Compute ask prices for this agent
                ask_prices = market.compute_ask_prices(agent, self.epsilon)
                
                # Find best bundle at current ask prices
                current_allocation = market.get_allocation(agent)
                best_bundle, best_surplus = agent.find_best_bundle(
                    market.slots, ask_prices, current_allocation
                )
                
                # Determine new slots to bid on
                new_slots = best_bundle - current_allocation
                
                # Record the round
                rounds.append(AuctionRound(
                    round_num=round_num,
                    bidder=agent,
                    slots_bid_on=new_slots,
                    allocations={
                        a.agent_id: sorted([s.slot_id for s in market.allocations.get(a.agent_id, frozenset())])
                        for a in market.agents
                    },
                    bid_prices=[market.bid_prices.get(s.slot_id, s.reserve_price) for s in market.slots]
                ))
                
                # Update bids and allocations for new slots
                for slot in new_slots:
                    # Increment bid price
                    market.bid_prices[slot.slot_id] = market.bid_prices[slot.slot_id] + self.epsilon
                    
                    # Remove from other agent if allocated
                    current_owner = market.get_slot_owner(slot)
                    if current_owner is not None and current_owner != agent:
                        owner_allocation = market.get_allocation(current_owner)
                        market.set_allocation(current_owner, owner_allocation - {slot})
                    
                    any_change = True
                
                # Update agent's allocation
                if best_bundle != current_allocation:
                    market.set_allocation(agent, best_bundle)
                    any_change = True
                
                if verbose and new_slots:
                    logger.info(
                        "Round %d: Agent %s bids on slots %s, prices: %s",
                        round_num, agent.name,
                        sorted([s.slot_id for s in new_slots]),
                        [market.bid_prices.get(s.slot_id, 0) for s in market.slots]
                    )
            
            iteration += 1
            
            # Check for convergence
            if not any_change:
                if verbose:
                    logger.info("Converged after %d iterations", iteration)
                break
        
        converged = iteration < self.max_iterations
        final_welfare = market.compute_welfare()
        
        return AuctionResult(
            market=market,
            rounds=rounds,
            converged=converged,
            iterations=iteration,
            final_welfare=final_welfare
        )
