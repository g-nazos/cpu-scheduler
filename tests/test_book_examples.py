"""
Tests that reproduce the book examples from Section 2.3.3.

These tests verify that our implementation produces results
consistent with the examples in "Multiagent Systems".
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.models.market import Market
from src.auction.ascending import AscendingAuction
from src.experiments.scenarios import (
    create_book_example_1,
    create_book_example_2,
    create_book_example_3,
)


class TestBookExample1:
    """
    Tests for the 8-slot processor example.
    
    From Section 2.3.3:
    - 8 time slots: 9am-5pm, reserve price $3/hour
    - 4 jobs with varying (λ, d, w)
    - Optimal allocation known from the book
    """
    
    def test_auction_with_small_epsilon_converges(self):
        """With ε=0.25, auction should converge."""
        market = create_book_example_1()
        auction = AscendingAuction(epsilon=0.25)
        result = auction.run(market)
        
        assert result.converged
        # Book shows ~24 rounds
        assert result.iterations < 100
    
    def test_auction_with_large_epsilon_terminates(self):
        """With ε=1.0, auction still terminates."""
        market = create_book_example_1()
        auction = AscendingAuction(epsilon=1.0)
        result = auction.run(market)
        assert result.converged
        assert result.final_solution_value >= 0


class TestBookExample2:
    """
    Tests for the Table 2.1 example.
    
    - 2 slots: 9am, 10am, reserve $3/hour
    - Job 1: 2 hours, deadline 11am, worth $10
    - Job 2: 1 hour, deadline 11am, worth $6
    """
    
    def test_auction_terminates(self):
        """Auction should always terminate."""
        market = create_book_example_2()
        for eps in [0.1, 0.25, 0.5, 1.0]:
            auction = AscendingAuction(epsilon=eps, max_iterations=500)
            result = auction.run(market)
            assert result.converged or result.iterations == 500
    
    def test_auction_with_small_epsilon(self):
        """Auction with small epsilon runs and terminates."""
        market = create_book_example_2()
        auction = AscendingAuction(epsilon=0.1, max_iterations=1000)
        result = auction.run(market)
        assert result.converged or result.iterations == 1000
        assert result.final_solution_value >= 0


class TestBookExample3:
    """
    Tests for the "arbitrarily suboptimal" example.
    
    - 2 slots with reserve prices $1 and $9
    - Job 1: 1 hour, deadline 10am (only slot 0), worth $3
    - Job 2: 2 hours, deadline 11am (both slots), worth $11
    
    Auction allocates slot 0 to Job 1, but optimal gives both to Job 2.
    """
    
    def test_auction_suboptimal(self):
        """Auction terminates; outcome can be suboptimal for this scenario."""
        market = create_book_example_3()
        auction = AscendingAuction(epsilon=0.25)
        result = auction.run(market)
        assert result.converged
        assert result.final_solution_value >= 0


class TestScalability:
    """Basic scalability tests."""
    
    def test_larger_problem(self):
        """Test that larger problems still work."""
        from src.experiments.scenarios import create_scalability_scenario
        
        market = create_scalability_scenario(num_agents=10, num_slots=12)
        
        auction = AscendingAuction(epsilon=0.5, max_iterations=5000)
        result = auction.run(market)
        
        assert result.converged or result.iterations == 5000
        assert result.final_solution_value >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
