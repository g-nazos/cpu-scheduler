from dataclasses import dataclass


@dataclass
class Slot:
    """
    Represents a discrete time slot in the scheduling problem.
    
    Each slot x_j has a reserve price q_j representing the minimum
    value the resource owner would accept for that slot.
    
    Attributes:
        slot_id: Unique identifier for the slot (0-indexed)
        time_label: Human-readable time label (e.g., "9:00 AM")
        reserve_price: Minimum price q_j for this slot
    """
    slot_id: int
    time_label: str
    reserve_price: float
    
    def __hash__(self) -> int:
        return hash(self.slot_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Slot):
            return False
        return self.slot_id == other.slot_id
    
    def __repr__(self) -> str:
        return f"Slot({self.slot_id}, '{self.time_label}', q={self.reserve_price})"
    
    def __lt__(self, other: "Slot") -> bool:
        """Enable sorting slots by ID (chronological order)."""
        return self.slot_id < other.slot_id


def create_slots(
    num_slots: int,
    reserve_price: float,
    start_hour: int = 9,
    slot_duration_hours: int = 1
) -> list[Slot]:
    """
    Create a list of consecutive time slots with uniform reserve price.
    
    Args:
        num_slots: Number of slots to create
        reserve_price: Reserve price for all slots
        start_hour: Starting hour (24-hour format)
        slot_duration_hours: Duration of each slot in hours
        
    Returns:
        List of Slot objects
    """
    slots = []
    for i in range(num_slots):
        hour = start_hour + i * slot_duration_hours
        if hour < 12:
            time_label = f"{hour}:00 AM"
        elif hour == 12:
            time_label = "12:00 PM"
        else:
            time_label = f"{hour - 12}:00 PM"
        
        slots.append(Slot(
            slot_id=i,
            time_label=time_label,
            reserve_price=reserve_price
        ))
    
    return slots


def create_slots_with_prices(
    reserve_prices: list[float],
    start_hour: int = 9,
    slot_duration_hours: int = 1
) -> list[Slot]:
    """
    Create slots with individual reserve prices.
    
    Args:
        reserve_prices: List of reserve prices for each slot
        start_hour: Starting hour (24-hour format)
        slot_duration_hours: Duration of each slot in hours
        
    Returns:
        List of Slot objects
    """
    slots = []
    for i, price in enumerate(reserve_prices):
        hour = start_hour + i * slot_duration_hours
        if hour < 12:
            time_label = f"{hour}:00 AM"
        elif hour == 12:
            time_label = "12:00 PM"
        else:
            time_label = f"{hour - 12}:00 PM"
        
        slots.append(Slot(
            slot_id=i,
            time_label=time_label,
            reserve_price=price
        ))
    
    return slots
