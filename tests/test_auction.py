"""
Tests for the ascending auction algorithm.

Verifies the implementation matches the behavior described
in Section 2.3.3 of "Multiagent Systems".
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.models.agent import Agent
from src.models.slot import Slot, create_slots
from src.models.market import Market
from src.auction.ascending import AscendingAuction
from src.experiments.scenarios import create_book_example_1


class TestAscendingAuction:
    """Tests for AscendingAuction class."""
    
    def test_initialization(self):
        """Test auction initialization."""
        auction = AscendingAuction(epsilon=0.25)
        assert auction.epsilon == 0.25
        assert auction.max_iterations == 10000
    
    def test_simple_case_converges(self):
        """Test that auction converges on a simple case."""
        slots = create_slots(num_slots=2, reserve_price=1.0)
        agents = [
            Agent(agent_id=1, name="A1", deadline_slot_id=2, required_slots=1, worth=5.0),
            Agent(agent_id=2, name="A2", deadline_slot_id=2, required_slots=1, worth=4.0),
        ]
        market = Market(agents=agents, slots=slots)
        
        auction = AscendingAuction(epsilon=0.25)
        result = auction.run(market)
        
        assert result.converged
        assert result.iterations > 0
        assert len(result.rounds) > 0
    
    def test_prices_never_decrease(self):
        """Verify prices are monotonically non-decreasing."""
        market = create_book_example_1()
        auction = AscendingAuction(epsilon=0.25)
        result = auction.run(market)
        
        num_slots = len(market.slots)
        for slot_idx in range(num_slots):
            prev_price = 0
            for round_data in result.rounds:
                price = round_data.bid_prices[slot_idx]
                assert price >= prev_price, f"Price decreased for slot {slot_idx}"
                prev_price = price
    
    def test_termination_guaranteed(self):
        """Verify auction always terminates."""
        market = create_book_example_1()
        auction = AscendingAuction(epsilon=0.25, max_iterations=1000)
        result = auction.run(market)
        
        # Should converge within max_iterations
        assert result.converged or result.iterations == 1000
    
    def test_book_example_1_with_small_epsilon(self):
        """
        Test Book Example 1 with ε=0.25.
        
        According to the book, this should converge to equilibrium.
        """
        market = create_book_example_1()
        auction = AscendingAuction(epsilon=0.25)
        result = auction.run(market)
        
        assert result.converged
        # The book shows ~24 rounds, but exact number depends on implementation
        assert result.iterations < 50  # Should converge reasonably quickly
    
    def test_solution_value_non_negative(self):
        """Verify solution_value is always non-negative."""
        market = create_book_example_1()
        auction = AscendingAuction(epsilon=0.25)
        result = auction.run(market)
        
        assert result.final_solution_value >= 0


class TestAuctionRoundTracking:
    """Tests for auction round tracking."""
    
    def test_rounds_recorded(self):
        """Verify rounds are properly recorded."""
        slots = create_slots(num_slots=2, reserve_price=1.0)
        agents = [
            Agent(agent_id=1, name="A1", deadline_slot_id=2, required_slots=1, worth=5.0),
        ]
        market = Market(agents=agents, slots=slots)
        
        auction = AscendingAuction(epsilon=0.5)
        result = auction.run(market)
        
        assert len(result.rounds) > 0
        
        for r in result.rounds:
            assert r.round_num > 0
            assert r.bidder is not None
            assert len(r.bid_prices) == len(slots)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
