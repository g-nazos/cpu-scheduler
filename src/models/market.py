"""
Market state representation for the scheduling problem.

Based on Section 2.3.3 of "Multiagent Systems" by Shoham & Leyton-Brown.
"""

from dataclasses import dataclass, field
from typing import FrozenSet, Optional
import copy

from src.models.agent import Agent
from src.models.slot import Slot


@dataclass
class Market:
    """
    Represents the market state in the scheduling problem.
    
    The market tracks:
    - All agents N and slots X
    - Current allocations F = (F_1, ..., F_n) where F_i is agent i's bundle
    - Current bid prices b = (b_1, ..., b_m) where b_j is slot j's bid
    
    Attributes:
        agents: List of all agents
        slots: List of all time slots
        allocations: Dict mapping agent_id to set of allocated slots
        bid_prices: Dict mapping slot_id to current bid price
    """
    agents: list[Agent]
    slots: list[Slot]
    allocations: dict[int, FrozenSet[Slot]] = field(default_factory=dict)
    bid_prices: dict[int, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize allocations and bid prices if not provided."""
        # Initialize empty allocations for all agents
        if not self.allocations:
            self.allocations = {agent.agent_id: frozenset() for agent in self.agents}
        
        # Initialize bid prices to reserve prices
        if not self.bid_prices:
            self.bid_prices = {slot.slot_id: slot.reserve_price for slot in self.slots}
    
    def get_allocation(self, agent: Agent) -> FrozenSet[Slot]:
        """Get the current allocation for an agent."""
        return self.allocations.get(agent.agent_id, frozenset())
    
    def set_allocation(self, agent: Agent, slots: FrozenSet[Slot]) -> None:
        """Set the allocation for an agent."""
        self.allocations[agent.agent_id] = slots
    
    def get_bid_price(self, slot: Slot) -> float:
        """Get the current bid price for a slot."""
        return self.bid_prices.get(slot.slot_id, slot.reserve_price)
    
    def set_bid_price(self, slot: Slot, price: float) -> None:
        """Set the bid price for a slot."""
        self.bid_prices[slot.slot_id] = price
    
    def get_unallocated_slots(self) -> FrozenSet[Slot]:
        """Get all slots that are not allocated to any agent (F_∅)."""
        allocated = set()
        for slots in self.allocations.values():
            allocated.update(slots)
        return frozenset(set(self.slots) - allocated)
    
    def get_slot_owner(self, slot: Slot) -> Optional[Agent]:
        """Get the agent who currently owns a slot, or None if unallocated."""
        for agent in self.agents:
            if slot in self.allocations.get(agent.agent_id, frozenset()):
                return agent
        return None
    
    def compute_ask_prices(self, agent: Agent, epsilon: float) -> dict[int, float]:
        """
        Compute ask prices for an agent.
        
        From Figure 2.7:
        - For slots agent already holds: p_j = b_j
        - For slots agent doesn't hold: p_j = b_j + ε
        
        Args:
            agent: The agent computing prices
            epsilon: The price increment
            
        Returns:
            Dictionary mapping slot_id to ask price
        """
        current_slots = self.get_allocation(agent)
        ask_prices = {}
        
        for slot in self.slots:
            if slot in current_slots:
                ask_prices[slot.slot_id] = self.bid_prices[slot.slot_id]
            else:
                ask_prices[slot.slot_id] = self.bid_prices[slot.slot_id] + epsilon
        
        return ask_prices
    
    def compute_solution_value(self) -> float:
        """
        Compute the total solution value V(F) of the current allocation.
        
        V(F) = Σ_{j|x_j ∈ F_∅} q_j + Σ_{i∈N} v_i(F_i)
        
        Returns:
            Total solution value
        """
        # Value from unallocated slots (reserve prices)
        unallocated = self.get_unallocated_slots()
        reserve_value = sum(slot.reserve_price for slot in unallocated)
        
        # Value from agent valuations
        agent_value = sum(
            agent.valuation(self.allocations.get(agent.agent_id, frozenset()))
            for agent in self.agents
        )
        
        return reserve_value + agent_value
    
    def copy(self) -> "Market":
        """Create a deep copy of the market state."""
        return Market(
            agents=self.agents,  # Agents are immutable, no need to copy
            slots=self.slots,    # Slots are immutable, no need to copy
            allocations=copy.deepcopy(self.allocations),
            bid_prices=copy.deepcopy(self.bid_prices)
        )
    
    def get_state_snapshot(self) -> dict:
        """
        Get a snapshot of the current market state for logging.
        
        Returns:
            Dictionary with allocation and price information
        """
        return {
            "allocations": {
                agent.name: sorted([s.slot_id for s in self.allocations.get(agent.agent_id, frozenset())])
                for agent in self.agents
            },
            "bid_prices": [self.bid_prices.get(s.slot_id, s.reserve_price) for s in self.slots],
            "solution_value": self.compute_solution_value()
        }
    
    def __repr__(self) -> str:
        lines = ["Market State:"]
        lines.append(f"  Agents: {len(self.agents)}, Slots: {len(self.slots)}")
        lines.append("  Allocations:")
        for agent in self.agents:
            slots = self.allocations.get(agent.agent_id, frozenset())
            slot_ids = sorted([s.slot_id for s in slots])
            lines.append(f"    {agent.name}: {slot_ids}")
        lines.append(f"  Bid Prices: {[self.bid_prices.get(s.slot_id, 0) for s in self.slots]}")
        lines.append(f"  Solution Value: {self.compute_solution_value():.2f}")
        return "\n".join(lines)
