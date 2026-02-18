import logging
from dataclasses import dataclass

from src.models.market import Market


logger = logging.getLogger(__name__)


@dataclass
class EquilibriumCheckResult:
    """Result of checking competitive equilibrium conditions."""
    is_equilibrium: bool
    surplus_maximization_satisfied: bool
    unallocated_price_satisfied: bool
    allocated_price_satisfied: bool
    violations: list[str]
    agent_surpluses: dict[int, float]  # agent_id -> surplus at current allocation
    best_surpluses: dict[int, float]   # agent_id -> best possible surplus


class EquilibriumChecker:
    """
    Checker for Competitive Equilibrium (Definition 2.3.11).
    
    A solution F is in competitive equilibrium at prices p iff:
    
    1. Surplus Maximization: For all i ∈ N:
       F_i = argmax_{T⊆X} (v_i(T) - Σ_{j|x_j∈T} p_j)
       
    2. Unallocated Slots: For all j where x_j ∈ F_∅: p_j = q_j
    
    3. Allocated Slots: For all j where x_j ∉ F_∅: p_j ≥ q_j
    """
    
    def __init__(self, market: Market, tolerance: float = 1e-6):
        """
        Initialize the checker.
        
        Args:
            market: Market state to check
            tolerance: Tolerance for floating point comparisons
        """
        self.market = market
        self.tolerance = tolerance
    
    def check(self) -> EquilibriumCheckResult:
        """
        Check if the current market state is a competitive equilibrium.
        
        Returns:
            EquilibriumCheckResult with detailed findings
        """
        violations = []
        agent_surpluses = {}
        best_surpluses = {}
        
        # Check condition 1: Surplus Maximization
        surplus_ok = True
        for agent in self.market.agents:
            current_allocation = self.market.get_allocation(agent)
            current_surplus = agent.surplus(current_allocation, self.market.bid_prices)
            agent_surpluses[agent.agent_id] = current_surplus
            
            # Find best possible bundle
            best_bundle, best_surplus = agent.find_best_bundle(
                self.market.slots,
                self.market.bid_prices,
                frozenset()  # Consider all bundles, not just extensions
            )
            best_surpluses[agent.agent_id] = best_surplus
            
            # Check if current allocation is optimal (within tolerance)
            if best_surplus > current_surplus + self.tolerance:
                surplus_ok = False
                violations.append(
                    f"Agent {agent.name}: current surplus {current_surplus:.2f} < "
                    f"best surplus {best_surplus:.2f} (bundle {sorted([s.slot_id for s in best_bundle])})"
                )
        
        # Check condition 2: Unallocated slots have price = reserve
        unallocated_ok = True
        unallocated_slots = self.market.get_unallocated_slots()
        for slot in unallocated_slots:
            price = self.market.bid_prices.get(slot.slot_id, slot.reserve_price)
            if abs(price - slot.reserve_price) > self.tolerance:
                unallocated_ok = False
                violations.append(
                    f"Unallocated slot {slot.slot_id}: price {price:.2f} != "
                    f"reserve {slot.reserve_price:.2f}"
                )
        
        # Check condition 3: Allocated slots have price >= reserve
        allocated_ok = True
        allocated_slots = set(self.market.slots) - set(unallocated_slots)
        for slot in allocated_slots:
            price = self.market.bid_prices.get(slot.slot_id, slot.reserve_price)
            if price < slot.reserve_price - self.tolerance:
                allocated_ok = False
                violations.append(
                    f"Allocated slot {slot.slot_id}: price {price:.2f} < "
                    f"reserve {slot.reserve_price:.2f}"
                )
        
        is_equilibrium = surplus_ok and unallocated_ok and allocated_ok
        
        return EquilibriumCheckResult(
            is_equilibrium=is_equilibrium,
            surplus_maximization_satisfied=surplus_ok,
            unallocated_price_satisfied=unallocated_ok,
            allocated_price_satisfied=allocated_ok,
            violations=violations,
            agent_surpluses=agent_surpluses,
            best_surpluses=best_surpluses
        )


def check_equilibrium(market: Market, tolerance: float = 1e-6) -> EquilibriumCheckResult:
    """
    Convenience function to check equilibrium for a market.
    
    Args:
        market: Market state to check
        tolerance: Tolerance for floating point comparisons
        
    Returns:
        EquilibriumCheckResult
    """
    checker = EquilibriumChecker(market, tolerance)
    return checker.check()


def print_equilibrium_report(result: EquilibriumCheckResult, market: Market) -> None:
    """
    Print a detailed equilibrium report.
    
    Args:
        result: The equilibrium check result
        market: The market that was checked
    """
    print("\n" + "=" * 60)
    print("COMPETITIVE EQUILIBRIUM CHECK")
    print("=" * 60)
    
    status = "YES" if result.is_equilibrium else "NO"
    print(f"\nIs Competitive Equilibrium: {status}")
    
    print(f"\nCondition 1 (Surplus Maximization): {'PASS' if result.surplus_maximization_satisfied else 'FAIL'}")
    print(f"Condition 2 (Unallocated Prices = Reserve): {'PASS' if result.unallocated_price_satisfied else 'FAIL'}")
    print(f"Condition 3 (Allocated Prices >= Reserve): {'PASS' if result.allocated_price_satisfied else 'FAIL'}")
    
    if result.violations:
        print("\nViolations:")
        for v in result.violations:
            print(f"  - {v}")
    
    print("\nAgent Surpluses:")
    for agent in market.agents:
        current = result.agent_surpluses.get(agent.agent_id, 0)
        best = result.best_surpluses.get(agent.agent_id, 0)
        gap = best - current
        print(f"  {agent.name}: current={current:.2f}, best={best:.2f}, gap={gap:.2f}")
    
    print("=" * 60)
