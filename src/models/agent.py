"""
Agent (job) representation for the scheduling problem.

Based on Definition 2.3.10 from Section 2.3.3 of "Multiagent Systems"
by Shoham & Leyton-Brown.
"""

from dataclasses import dataclass
from typing import FrozenSet, Set

from src.models.slot import Slot


@dataclass
class Agent:
    """
    Represents an agent (job) in the scheduling problem.
    
    Each agent i has:
    - A deadline d_i: the latest slot by which the job must complete
    - A required number of slots λ_i: how many time slots the job needs
    - A worth w_i: the value if the job completes on time
    
    The valuation function v_i(F_i) is defined as:
        v_i(F_i) = w_i  if F_i includes λ_i slots before d_i
                 = 0    otherwise
    
    Attributes:
        agent_id: Unique identifier for the agent
        name: Human-readable name for the agent
        deadline_slot_id: The slot ID representing the deadline (exclusive)
        required_slots: Number of slots needed (λ_i)
        worth: Value of completing the job (w_i)
    """
    agent_id: int
    name: str
    deadline_slot_id: int  # Slots with id < deadline_slot_id are before deadline
    required_slots: int    # λ_i
    worth: float           # w_i
    
    def __hash__(self) -> int:
        return hash(self.agent_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Agent):
            return False
        return self.agent_id == other.agent_id
    
    def __repr__(self) -> str:
        return (f"Agent({self.agent_id}, '{self.name}', "
                f"d={self.deadline_slot_id}, λ={self.required_slots}, w={self.worth})")
    
    def valuation(self, allocated_slots: Set[Slot] | FrozenSet[Slot]) -> float:
        """
        Compute the valuation v_i(F_i) for a given bundle of slots.
        
        v_i(F_i) = w_i  if F_i includes λ_i slots before deadline d_i
                 = 0    otherwise
        
        Args:
            allocated_slots: Set of slots allocated to this agent
            
        Returns:
            The valuation (worth if satisfied, 0 otherwise)
        """
        if not allocated_slots:
            return 0.0
        
        # Count slots that are before the deadline
        slots_before_deadline = sum(
            1 for slot in allocated_slots
            if slot.slot_id < self.deadline_slot_id
        )
        
        if slots_before_deadline >= self.required_slots:
            return self.worth
        return 0.0
    
    def surplus(
        self,
        allocated_slots: Set[Slot] | FrozenSet[Slot],
        prices: dict[int, float]
    ) -> float:
        """
        Compute the surplus for a given bundle at given prices.
        
        Surplus = v_i(F_i) - Σ_{j ∈ F_i} p_j
        
        Args:
            allocated_slots: Set of slots allocated to this agent
            prices: Dictionary mapping slot_id to price
            
        Returns:
            The surplus value
        """
        value = self.valuation(allocated_slots)
        cost = sum(prices.get(slot.slot_id, 0.0) for slot in allocated_slots)
        return value - cost
    
    def get_valid_slots(self, all_slots: list[Slot]) -> list[Slot]:
        """
        Get all slots that are before this agent's deadline.
        
        Args:
            all_slots: List of all available slots
            
        Returns:
            List of slots that are before the deadline
        """
        return [slot for slot in all_slots if slot.slot_id < self.deadline_slot_id]
    
    def find_best_bundle(
        self,
        all_slots: list[Slot],
        prices: dict[int, float],
        current_allocation: Set[Slot] | FrozenSet[Slot]
    ) -> tuple[FrozenSet[Slot], float]:
        """
        Find the bundle that maximizes surplus at given prices.
        
        This implements the argmax in the ascending auction:
        S* = argmax_{S⊆X, S⊇F_i} (v_i(S) - Σ_{j∈S} p_j)
        
        For the scheduling problem, we need to find the cheapest
        set of required_slots slots before the deadline, or the empty set.
        
        Args:
            all_slots: List of all available slots
            prices: Dictionary mapping slot_id to price
            current_allocation: Current slots held by this agent
            
        Returns:
            Tuple of (best_bundle, best_surplus)
        """
        valid_slots = self.get_valid_slots(all_slots)
        
        # If not enough slots available before deadline, return empty
        if len(valid_slots) < self.required_slots:
            return frozenset(), 0.0
        
        # Sort valid slots by price (ascending)
        sorted_slots = sorted(valid_slots, key=lambda s: prices.get(s.slot_id, 0.0))
        
        # The best bundle is the cheapest required_slots slots
        best_bundle = frozenset(sorted_slots[:self.required_slots])
        best_surplus = self.surplus(best_bundle, prices)
        
        # Compare with empty bundle (surplus = 0)
        if best_surplus < 0:
            return frozenset(), 0.0
        
        return best_bundle, best_surplus
