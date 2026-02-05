"""
Integer Program Solver for the Scheduling Problem.

Solves the optimal scheduling problem using integer programming
as the baseline for comparison with the auction algorithm.

Based on the IP formulation in Section 2.3.3 of "Multiagent Systems"
by Shoham & Leyton-Brown.
"""

import itertools
import logging
from dataclasses import dataclass
from typing import FrozenSet, Optional

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from src.models.agent import Agent
from src.models.slot import Slot
from src.models.market import Market


logger = logging.getLogger(__name__)


@dataclass
class IPSolution:
    """Solution from the integer program solver."""
    allocations: dict[int, FrozenSet[Slot]]  # agent_id -> allocated slots
    optimal_welfare: float
    solved: bool
    solver_message: str


class IntegerProgramSolver:
    """
    Integer Program Solver for the Scheduling Problem.
    
    Formulation:
        maximize    Σ_{S⊆X, i∈N} v_i(S) · x_{i,S}
        
        subject to  Σ_{S⊆X} x_{i,S} ≤ 1              ∀i ∈ N
                    Σ_{S⊆X: j∈S, i∈N} x_{i,S} ≤ 1    ∀j ∈ X
                    x_{i,S} ∈ {0, 1}
    
    Note: For efficiency, we only consider bundles that give non-zero valuation.
    """
    
    def __init__(self, market: Market):
        """
        Initialize the solver with a market.
        
        Args:
            market: The market to solve
        """
        self.market = market
        self.agents = market.agents
        self.slots = market.slots
        
        # Generate valid bundles for each agent
        self._generate_bundles()
    
    def _generate_bundles(self) -> None:
        """Generate all valid bundles for each agent."""
        self.bundles: dict[int, list[FrozenSet[Slot]]] = {}
        self.bundle_values: dict[int, list[float]] = {}
        
        for agent in self.agents:
            valid_slots = agent.get_valid_slots(self.slots)
            agent_bundles = []
            agent_values = []
            
            # Only consider bundles of exactly required_slots size
            # (smaller bundles have 0 value, larger bundles are wasteful)
            if len(valid_slots) >= agent.required_slots:
                for combo in itertools.combinations(valid_slots, agent.required_slots):
                    bundle = frozenset(combo)
                    value = agent.valuation(bundle)
                    if value > 0:
                        agent_bundles.append(bundle)
                        agent_values.append(value)
            
            # Also consider empty bundle (always valid with value 0)
            agent_bundles.append(frozenset())
            agent_values.append(0.0)
            
            self.bundles[agent.agent_id] = agent_bundles
            self.bundle_values[agent.agent_id] = agent_values
            
            logger.debug(
                "Agent %s has %d valid bundles",
                agent.name, len(agent_bundles)
            )
    
    def solve(self) -> IPSolution:
        """
        Solve the integer program to find the optimal allocation.
        
        Returns:
            IPSolution with optimal allocations and welfare
        """
        # For small problems, use brute force enumeration
        # (scipy.milp can be slow for many binary variables)
        return self._solve_brute_force()
    
    def _solve_brute_force(self) -> IPSolution:
        """
        Solve by enumerating all valid allocations.
        
        For small problems, this is fast and guaranteed optimal.
        """
        best_welfare = float("-inf")
        best_allocation: dict[int, FrozenSet[Slot]] = {}
        
        # Get bundle indices for each agent
        agent_ids = [a.agent_id for a in self.agents]
        bundle_indices = [range(len(self.bundles[aid])) for aid in agent_ids]
        
        # Enumerate all combinations
        for combo in itertools.product(*bundle_indices):
            # Check if allocation is valid (no slot assigned twice)
            used_slots: set[Slot] = set()
            valid = True
            allocation: dict[int, FrozenSet[Slot]] = {}
            
            for agent_idx, bundle_idx in enumerate(combo):
                agent_id = agent_ids[agent_idx]
                bundle = self.bundles[agent_id][bundle_idx]
                
                # Check for conflicts
                if used_slots & bundle:
                    valid = False
                    break
                
                used_slots.update(bundle)
                allocation[agent_id] = bundle
            
            if not valid:
                continue
            
            # Compute welfare
            welfare = self._compute_welfare(allocation)
            
            if welfare > best_welfare:
                best_welfare = welfare
                best_allocation = allocation
        
        return IPSolution(
            allocations=best_allocation,
            optimal_welfare=best_welfare,
            solved=True,
            solver_message="Solved by brute force enumeration"
        )
    
    def _compute_welfare(self, allocation: dict[int, FrozenSet[Slot]]) -> float:
        """
        Compute welfare for an allocation.
        
        V(F) = Σ_{j|x_j ∈ F_∅} q_j + Σ_{i∈N} v_i(F_i)
        """
        # Compute allocated slots
        allocated: set[Slot] = set()
        for bundle in allocation.values():
            allocated.update(bundle)
        
        # Reserve value from unallocated slots
        unallocated = set(self.slots) - allocated
        reserve_value = sum(slot.reserve_price for slot in unallocated)
        
        # Agent valuations
        agent_value = 0.0
        for agent in self.agents:
            bundle = allocation.get(agent.agent_id, frozenset())
            agent_value += agent.valuation(bundle)
        
        return reserve_value + agent_value


def compute_optimal_welfare(market: Market) -> tuple[float, dict[int, FrozenSet[Slot]]]:
    """
    Convenience function to compute optimal welfare for a market.
    
    Args:
        market: The market to solve
        
    Returns:
        Tuple of (optimal_welfare, optimal_allocations)
    """
    solver = IntegerProgramSolver(market)
    solution = solver.solve()
    return solution.optimal_welfare, solution.allocations
