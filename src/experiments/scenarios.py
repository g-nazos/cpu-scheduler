"""
Experimental Scenarios from Section 2.3.3.

Creates the book examples and custom scenarios for testing
the ascending auction algorithm.
"""

from src.models.agent import Agent
from src.models.slot import Slot, create_slots, create_slots_with_prices
from src.models.market import Market


def create_book_example_1() -> Market:
    """
    Create the main 8-slot processor example from Section 2.3.3.
    
    8 time slots: 9am-5pm, reserve price $3/hour
    4 jobs:
        Job 1: 2 hours, deadline 1:00 PM, worth $10.00
        Job 2: 2 hours, deadline 12:00 PM, worth $16.00
        Job 3: 1 hour, deadline 12:00 PM, worth $6.00
        Job 4: 4 hours, deadline 5:00 PM, worth $14.50
    
    Expected optimal allocation:
        9:00 AM  -> Agent 2
        10:00 AM -> Agent 2
        11:00 AM -> Agent 1
        12:00 PM -> Agent 1
        1:00 PM  -> Agent 4
        2:00 PM  -> Agent 4
        3:00 PM  -> Agent 4
        4:00 PM  -> Agent 4
    
    With ε=0.25: converges in ~24 rounds
    With ε=1.00: may not converge or take many rounds
    """
    # Create 8 slots from 9am to 5pm with reserve price $3
    slots = create_slots(num_slots=8, reserve_price=3.0, start_hour=9)
    
    # Create agents (jobs)
    # Deadline slot IDs: 12pm = slot 3 (0-indexed), 1pm = slot 4, 5pm = slot 8
    agents = [
        Agent(
            agent_id=1,
            name="Job1",
            deadline_slot_id=4,  # Before 1:00 PM means slots 0,1,2,3 (9am-12pm)
            required_slots=2,
            worth=10.0
        ),
        Agent(
            agent_id=2,
            name="Job2",
            deadline_slot_id=3,  # Before 12:00 PM means slots 0,1,2 (9am-11am)
            required_slots=2,
            worth=16.0
        ),
        Agent(
            agent_id=3,
            name="Job3",
            deadline_slot_id=3,  # Before 12:00 PM means slots 0,1,2 (9am-11am)
            required_slots=1,
            worth=6.0
        ),
        Agent(
            agent_id=4,
            name="Job4",
            deadline_slot_id=8,  # Before 5:00 PM means all slots
            required_slots=4,
            worth=14.5
        ),
    ]
    
    return Market(agents=agents, slots=slots)


def create_book_example_1_two_cpus() -> Market:
    """
    Same as book example 1 but with 2 identical CPUs (16 slots total).
    
    8 time slots × 2 CPUs: slot_ids 0–7 = CPU 0, 8–15 = CPU 1 (same times).
    Jobs are unchanged; each job can be assigned a consecutive block on either CPU.
    """
    slots = create_slots(num_slots=8, reserve_price=3.0, start_hour=9, num_cpus=2)
    agents = [
        Agent(agent_id=1, name="Job1", deadline_slot_id=4, required_slots=2, worth=10.0),
        Agent(agent_id=2, name="Job2", deadline_slot_id=3, required_slots=2, worth=16.0),
        Agent(agent_id=3, name="Job3", deadline_slot_id=3, required_slots=1, worth=6.0),
        Agent(agent_id=4, name="Job4", deadline_slot_id=8, required_slots=4, worth=14.5),
    ]
    return Market(agents=agents, slots=slots)


def create_duplicate_example_1(num_cpus: int = 1) -> Market:
    """
    Exactly duplicate jobs of book example 1: same 4 job types, each repeated once (8 jobs total).
    Same 8 time slots (or 16 with 2 CPUs). Job5–Job8 have same (deadline, length, worth) as Job1–Job4.
    """
    slots = create_slots(num_slots=8, reserve_price=3.0, start_hour=9, num_cpus=num_cpus)
    agents = [
        Agent(agent_id=1, name="2SlotsDead4", deadline_slot_id=4, required_slots=2, worth=10.0),
        Agent(agent_id=2, name="2SlotsDead3", deadline_slot_id=3, required_slots=2, worth=16.0),
        Agent(agent_id=3, name="1SlotsDead3", deadline_slot_id=3, required_slots=1, worth=6.0),
        Agent(agent_id=4, name="4SlotsDead8", deadline_slot_id=8, required_slots=4, worth=14.5),
        Agent(agent_id=5, name="2SlotsDead4_2", deadline_slot_id=4, required_slots=2, worth=10.0),
        Agent(agent_id=6, name="2SlotsDead3_2", deadline_slot_id=3, required_slots=2, worth=16.0),
        Agent(agent_id=7, name="1SlotsDead3_2", deadline_slot_id=3, required_slots=1, worth=6.0),
        Agent(agent_id=8, name="4SlotsDead8_2", deadline_slot_id=8, required_slots=4, worth=14.5),
    ]
    
    return Market(agents=agents, slots=slots)


def create_book_example_2() -> Market:
    """
    Create the Table 2.1 example (2 slots, 2 jobs).
    
    2 slots: 9am, 10am, reserve $3/hour
    2 jobs:
        Job 1: 2 hours, deadline 11:00 AM, worth $10.00
        Job 2: 1 hour, deadline 11:00 AM, worth $6.00
    
    The book notes that competitive equilibrium may not exist here due to complementarity.
    """
    slots = create_slots(num_slots=2, reserve_price=3.0, start_hour=9)
    
    agents = [
        Agent(
            agent_id=1,
            name="Job1",
            deadline_slot_id=2,  # Both slots are before deadline
            required_slots=2,
            worth=10.0
        ),
        Agent(
            agent_id=2,
            name="Job2",
            deadline_slot_id=2,  # Both slots are before deadline
            required_slots=1,
            worth=6.0
        ),
    ]
    
    return Market(agents=agents, slots=slots)


def create_book_example_2_two_cpus() -> Market:
    """
    Same as book example 2 but with 2 identical CPUs (4 slots: 9am and 10am on each CPU).
    Same 2 jobs; each can be assigned a consecutive block on either CPU.
    """
    slots = create_slots(num_slots=2, reserve_price=3.0, start_hour=9, num_cpus=2)
    agents = [
        Agent(agent_id=1, name="Job1", deadline_slot_id=2, required_slots=2, worth=10.0),
        Agent(agent_id=2, name="Job2", deadline_slot_id=2, required_slots=1, worth=6.0),
    ]
    return Market(agents=agents, slots=slots)


def create_book_example_3() -> Market:
    """
    Create the "arbitrarily suboptimal" example.
    
    2 slots with reserve prices $1 and $9
    2 jobs:
        Job 1: 1 hour, deadline 10:00 AM, worth $3.00
        Job 2: 2 hours, deadline 11:00 AM, worth $11.00
    
    The auction will stop with slot 0 -> Job 1, slot 1 -> Job 2
    but optimal would allocate both slots to Job 2.
    """
    slots = create_slots_with_prices(
        reserve_prices=[1.0, 9.0],
        start_hour=9
    )
    
    agents = [
        Agent(
            agent_id=1,
            name="Job1",
            deadline_slot_id=1,  # Only slot 0 is before deadline
            required_slots=1,
            worth=3.0
        ),
        Agent(
            agent_id=2,
            name="Job2",
            deadline_slot_id=2,  # Both slots are before deadline
            required_slots=2,
            worth=11.0
        ),
    ]
    
    return Market(agents=agents, slots=slots)


def create_many_jobs_example(
    num_slots: int = 8,
    reserve_price: float = 3.0,
    num_cpus: int = 1,
) -> Market:
    """
    Example with many jobs (8) competing for 8 time slots (or 16 with 2 CPUs).
    
    8 time slots: 9am-5pm, reserve $3/hour. Jobs have mixed deadlines,
    required lengths (1–3 slots), and worths so the auction has real competition.
    """
    slots = create_slots(num_slots=num_slots, reserve_price=reserve_price, start_hour=9, num_cpus=num_cpus)
    agents = [
        Agent(agent_id=1, name="Job1", deadline_slot_id=3, required_slots=2, worth=12.0),   # by 12pm, 2h
        Agent(agent_id=2, name="Job2", deadline_slot_id=3, required_slots=1, worth=8.0),   # by 12pm, 1h
        Agent(agent_id=3, name="Job3", deadline_slot_id=4, required_slots=2, worth=14.0),   # by 1pm, 2h
        Agent(agent_id=4, name="Job4", deadline_slot_id=4, required_slots=1, worth=5.0),    # by 1pm, 1h
        Agent(agent_id=5, name="Job5", deadline_slot_id=6, required_slots=2, worth=11.0),   # by 3pm, 2h
        Agent(agent_id=6, name="Job6", deadline_slot_id=6, required_slots=1, worth=7.0),   # by 3pm, 1h
        Agent(agent_id=7, name="Job7", deadline_slot_id=8, required_slots=3, worth=18.0),  # by 5pm, 3h
        Agent(agent_id=8, name="Job8", deadline_slot_id=8, required_slots=2, worth=10.0),  # by 5pm, 2h
    ]
    return Market(agents=agents, slots=slots)


def create_single_slot_demand_scenario(
    num_agents: int = 5,
    num_slots: int = 5,
    reserve_price: float = 1.0,
    max_worth: float = 10.0
) -> Market:
    """
    Create a scenario where each agent demands only 1 slot.
    
    According to Theorem 2.3.14, under these conditions a competitive equilibrium exists.
    
    Args:
        num_agents: Number of agents
        num_slots: Number of time slots
        reserve_price: Reserve price for all slots
        max_worth: Maximum worth for agents
    """
    import random
    
    slots = create_slots(num_slots=num_slots, reserve_price=reserve_price)
    
    agents = []
    for i in range(num_agents):
        # Each agent needs exactly 1 slot
        # Deadline is random but ensures at least some valid slots
        deadline = random.randint(1, num_slots)
        worth = random.uniform(reserve_price + 1, max_worth)
        
        agents.append(Agent(
            agent_id=i + 1,
            name=f"Job{i + 1}",
            deadline_slot_id=deadline,
            required_slots=1,
            worth=round(worth, 2)
        ))
    
    return Market(agents=agents, slots=slots)


def create_competitive_scenario() -> Market:
    """
    Create a scenario with high competition for early slots.
    
    Multiple agents want the same early slots, leading to
    competitive bidding and price escalation.
    """
    slots = create_slots(num_slots=4, reserve_price=2.0, start_hour=9)
    
    agents = [
        Agent(agent_id=1, name="Urgent1", deadline_slot_id=2, required_slots=1, worth=15.0),
        Agent(agent_id=2, name="Urgent2", deadline_slot_id=2, required_slots=1, worth=14.0),
        Agent(agent_id=3, name="Urgent3", deadline_slot_id=2, required_slots=1, worth=13.0),
        Agent(agent_id=4, name="Flex", deadline_slot_id=4, required_slots=2, worth=20.0),
    ]
    
    return Market(agents=agents, slots=slots)


def create_scalability_scenario(
    num_agents: int,
    num_slots: int,
    reserve_price: float = 1.0
) -> Market:
    """
    Create a scenario for scalability testing.
    
    Args:
        num_agents: Number of agents
        num_slots: Number of time slots
        reserve_price: Reserve price for all slots
    """
    import random
    
    slots = create_slots(num_slots=num_slots, reserve_price=reserve_price)
    
    agents = []
    for i in range(num_agents):
        # Random required slots (1 to 3)
        required = random.randint(1, min(3, num_slots))
        # Deadline ensuring enough slots available
        deadline = random.randint(required, num_slots)
        worth = random.uniform(reserve_price * required + 1, reserve_price * required + 10)
        
        agents.append(Agent(
            agent_id=i + 1,
            name=f"Job{i + 1}",
            deadline_slot_id=deadline,
            required_slots=required,
            worth=round(worth, 2)
        ))
    
    return Market(agents=agents, slots=slots)


# Dictionary mapping scenario names to creator functions
SCENARIOS = {
    "book_example_1": create_book_example_1,
    "book_example_2": create_book_example_2,
    "book_example_3": create_book_example_3,
    "single_slot_demand": create_single_slot_demand_scenario,
    "competitive": create_competitive_scenario,
    "scalability": create_scalability_scenario,
    "duplicate_example_1": create_duplicate_example_1,
    "book_example_1_two_cpus": create_book_example_1_two_cpus,
    "book_example_2_two_cpus": create_book_example_2_two_cpus,
    "many_jobs_example": create_many_jobs_example,
}
