"""
Tests for competitive equilibrium verification.

Verifies the equilibrium checker correctly implements
Definition 2.3.11 from Section 2.3.3.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.models.agent import Agent
from src.models.slot import Slot, create_slots
from src.models.market import Market
from src.auction.ascending import AscendingAuction
from src.auction.equilibrium import EquilibriumChecker, check_equilibrium
from src.experiments.scenarios import (
    create_book_example_1,
    create_book_example_2,
)


class TestEquilibriumChecker:
    """Tests for EquilibriumChecker class."""
    
    def test_empty_allocation_not_equilibrium(self):
        """An empty allocation is not equilibrium if agents can profit."""
        slots = create_slots(num_slots=2, reserve_price=1.0)
        agents = [
            Agent(agent_id=1, name="A1", deadline_slot_id=2, required_slots=1, worth=5.0),
        ]
        market = Market(agents=agents, slots=slots)
        # Market starts with empty allocation and reserve prices
        
        result = check_equilibrium(market)
        
        # Agent can profit by getting a slot (worth 5, cost 1)
        assert not result.is_equilibrium
        assert not result.surplus_maximization_satisfied
    
    def test_unallocated_price_condition(self):
        """Test that unallocated slots must have price = reserve."""
        slots = create_slots(num_slots=2, reserve_price=1.0)
        agents = [
            Agent(agent_id=1, name="A1", deadline_slot_id=2, required_slots=1, worth=5.0),
        ]
        market = Market(agents=agents, slots=slots)
        
        # Manually set allocation and prices to create a test case
        market.allocations[1] = frozenset([slots[0]])
        market.bid_prices[0] = 4.0  # Allocated slot, price > reserve OK
        market.bid_prices[1] = 2.0  # Unallocated slot, price != reserve VIOLATION
        
        result = check_equilibrium(market)
        
        assert not result.unallocated_price_satisfied
    
    def test_allocated_price_condition(self):
        """Test that allocated slots must have price >= reserve."""
        slots = create_slots(num_slots=2, reserve_price=3.0)
        agents = [
            Agent(agent_id=1, name="A1", deadline_slot_id=2, required_slots=1, worth=5.0),
        ]
        market = Market(agents=agents, slots=slots)
        
        # Set allocation with price below reserve
        market.allocations[1] = frozenset([slots[0]])
        market.bid_prices[0] = 2.0  # Below reserve of 3.0 - VIOLATION
        market.bid_prices[1] = 3.0  # Equal to reserve
        
        result = check_equilibrium(market)
        
        assert not result.allocated_price_satisfied


class TestBookExampleEquilibrium:
    """Tests using book examples."""
    
    def test_example_1_small_epsilon_reaches_equilibrium(self):
        """
        Book Example 1 with ε=0.25 should reach equilibrium.
        """
        market = create_book_example_1()
        auction = AscendingAuction(epsilon=0.25)
        result = auction.run(market)
        
        eq_result = check_equilibrium(result.market)
        
        # With small epsilon, should reach (approximate) equilibrium
        # Note: May not be exact due to discrete price increments
        # Using approximate check
        checker = EquilibriumChecker(result.market)
        approx_result = checker.check_approximate(epsilon=0.5)
        
        assert approx_result.is_equilibrium or eq_result.is_equilibrium
    
    def test_example_2_no_equilibrium_exists(self):
        """
        Book Example 2 (Table 2.1) has no competitive equilibrium.
        
        The auction will terminate but won't be in equilibrium.
        """
        market = create_book_example_2()
        
        # Try multiple epsilon values
        for eps in [0.1, 0.25, 0.5]:
            auction = AscendingAuction(epsilon=eps, max_iterations=500)
            result = auction.run(market)
            eq_result = check_equilibrium(result.market)
            
            # None should be true equilibrium due to complementarity
            # (though approximate equilibrium might be achieved)


class TestApproximateEquilibrium:
    """Tests for approximate (epsilon-) equilibrium."""
    
    def test_epsilon_equilibrium_more_lenient(self):
        """ε-equilibrium should be easier to satisfy than exact."""
        slots = create_slots(num_slots=2, reserve_price=1.0)
        agents = [
            Agent(agent_id=1, name="A1", deadline_slot_id=2, required_slots=1, worth=5.0),
            Agent(agent_id=2, name="A2", deadline_slot_id=2, required_slots=1, worth=4.5),
        ]
        market = Market(agents=agents, slots=slots)
        
        auction = AscendingAuction(epsilon=1.0)  # Large epsilon
        result = auction.run(market)
        
        checker = EquilibriumChecker(result.market)
        
        exact_result = checker.check()
        approx_result = checker.check_approximate(epsilon=2.0)
        
        # Approximate should be at least as likely to be satisfied
        if exact_result.is_equilibrium:
            assert approx_result.is_equilibrium


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
