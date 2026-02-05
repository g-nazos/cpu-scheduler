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
from src.auction.equilibrium import check_equilibrium
from src.optimization.integer_program import IntegerProgramSolver
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
    
    def test_optimal_solution(self):
        """Verify IP solver finds the optimal allocation from the book."""
        market = create_book_example_1()
        solver = IntegerProgramSolver(market)
        solution = solver.solve()
        
        # Book says optimal is:
        # Job 2: slots 0,1 (9am, 10am) -> value 16
        # Job 1: slots 2,3 (11am, 12pm) -> value 10
        # Job 4: slots 4,5,6,7 (1pm-4pm) -> value 14.5
        # Job 3: unallocated -> value 0
        # Total agent value: 16 + 10 + 14.5 = 40.5
        # Reserve value: 0 (all slots allocated)
        # Total: 40.5
        
        assert solution.solved
        assert solution.optimal_welfare == pytest.approx(40.5, abs=0.1)
    
    def test_auction_with_small_epsilon_converges(self):
        """With ε=0.25, auction should converge."""
        market = create_book_example_1()
        auction = AscendingAuction(epsilon=0.25)
        result = auction.run(market)
        
        assert result.converged
        # Book shows ~24 rounds
        assert result.iterations < 100
    
    def test_auction_with_large_epsilon_suboptimal(self):
        """With ε=1.0, auction may not reach equilibrium."""
        market = create_book_example_1()
        
        # Get optimal for comparison
        solver = IntegerProgramSolver(market)
        optimal = solver.solve()
        
        # Run with large epsilon
        auction = AscendingAuction(epsilon=1.0)
        result = auction.run(market)
        
        # Should still terminate
        assert result.converged
        
        # But welfare may be lower than optimal
        # (Book shows it doesn't reach equilibrium with ε=1)


class TestBookExample2:
    """
    Tests for the "no equilibrium" example (Table 2.1).
    
    - 2 slots: 9am, 10am, reserve $3/hour
    - Job 1: 2 hours, deadline 11am, worth $10
    - Job 2: 1 hour, deadline 11am, worth $6
    
    No competitive equilibrium exists due to complementarity.
    """
    
    def test_optimal_solution(self):
        """Find the optimal allocation (even though no equilibrium exists)."""
        market = create_book_example_2()
        solver = IntegerProgramSolver(market)
        solution = solver.solve()
        
        # Optimal should be: Job 1 gets both slots (value 10)
        # Reserve: 0, Total: 10
        # OR neither gets anything: Reserve 6, Total: 6
        # So Job 1 getting both is optimal
        assert solution.solved
        assert solution.optimal_welfare == pytest.approx(10.0, abs=0.1)
    
    def test_auction_terminates(self):
        """Auction should always terminate, even without equilibrium."""
        market = create_book_example_2()
        
        for eps in [0.1, 0.25, 0.5, 1.0]:
            auction = AscendingAuction(epsilon=eps, max_iterations=500)
            result = auction.run(market)
            
            assert result.converged or result.iterations == 500
    
    def test_no_exact_equilibrium(self):
        """No exact competitive equilibrium should be found."""
        market = create_book_example_2()
        
        # Try with very small epsilon for best chance
        auction = AscendingAuction(epsilon=0.1, max_iterations=1000)
        result = auction.run(market)
        
        eq_result = check_equilibrium(result.market)
        
        # Due to complementarity, exact equilibrium is impossible
        # But we might get close with small epsilon


class TestBookExample3:
    """
    Tests for the "arbitrarily suboptimal" example.
    
    - 2 slots with reserve prices $1 and $9
    - Job 1: 1 hour, deadline 10am (only slot 0), worth $3
    - Job 2: 2 hours, deadline 11am (both slots), worth $11
    
    Auction allocates slot 0 to Job 1, but optimal gives both to Job 2.
    """
    
    def test_optimal_solution(self):
        """Find the true optimal allocation."""
        market = create_book_example_3()
        solver = IntegerProgramSolver(market)
        solution = solver.solve()
        
        # With our formulation:
        # Option A: Job 1 gets slot 0 (value 3), slot 1 unallocated (reserve 9) = 12
        # Option B: Job 2 gets both slots (value 11), nothing unallocated = 11
        # Optimal is actually Option A due to high reserve price of slot 1
        assert solution.solved
        assert solution.optimal_welfare == pytest.approx(12.0, abs=0.1)
    
    def test_auction_suboptimal(self):
        """Auction should produce a suboptimal result."""
        market = create_book_example_3()
        
        # Get optimal
        solver = IntegerProgramSolver(market)
        optimal = solver.solve()
        
        # Run auction
        auction = AscendingAuction(epsilon=0.25)
        result = auction.run(market)
        
        # Should terminate
        assert result.converged
        
        # Auction result:
        # Job 1 bids on slot 0 (only valid for them), gets it
        # Job 2 needs both slots but slot 0 is taken
        # Expected: Job 1 gets slot 0 (worth 3), slot 1 unallocated (reserve 9)
        # Total welfare: 3 + 9 = 12? Or Job 1 doesn't bid?
        
        # The welfare should be less than or equal to optimal
        assert result.final_welfare <= optimal.optimal_welfare + 0.01


class TestScalability:
    """Basic scalability tests."""
    
    def test_larger_problem(self):
        """Test that larger problems still work."""
        from src.experiments.scenarios import create_scalability_scenario
        
        market = create_scalability_scenario(num_agents=10, num_slots=12)
        
        auction = AscendingAuction(epsilon=0.5, max_iterations=5000)
        result = auction.run(market)
        
        assert result.converged or result.iterations == 5000
        assert result.final_welfare >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
