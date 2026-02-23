from dataclasses import dataclass


@dataclass
class Slot:
    """
    Represents a discrete time slot in the scheduling problem.
    
    Each slot x_j has a reserve price q_j representing the minimum
    value the resource owner would accept for that slot.
    
    For multi-CPU: time_index is the time step (0..T-1), cpu_id is which CPU (0, 1, ...).
    For single-CPU (default): time_index and cpu_id are omitted; slot_id is used as time index.
    
    Attributes:
        slot_id: Unique identifier for the slot (0-indexed)
        time_label: Human-readable time label (e.g., "9:00 AM")
        reserve_price: Minimum price q_j for this slot
        time_index: Time step index (for multi-CPU); if None, slot_id is used (single-CPU)
        cpu_id: Which CPU this slot belongs to (0, 1, ...); 0 for single-CPU
    """
    slot_id: int
    time_label: str
    reserve_price: float
    time_index: int | None = None
    cpu_id: int = 0
    
    def __hash__(self) -> int:
        return hash(self.slot_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Slot):
            return False
        return self.slot_id == other.slot_id
    
    def __repr__(self) -> str:
        extra = f", t={self.time_index}, cpu={self.cpu_id}" if self.time_index is not None else ""
        return f"Slot({self.slot_id}, '{self.time_label}', q={self.reserve_price}{extra})"

    def get_time_index(self) -> int:
        """Time step for this slot (slot_id when single-CPU)."""
        return self.slot_id if self.time_index is None else self.time_index

    def __lt__(self, other: "Slot") -> bool:
        """Enable sorting slots by ID (chronological order)."""
        return self.slot_id < other.slot_id


def _time_label(hour: int) -> str:
    """Format hour as 9:00 AM / 12:00 PM / 1:00 PM."""
    if hour < 12:
        return f"{hour}:00 AM"
    if hour == 12:
        return "12:00 PM"
    return f"{hour - 12}:00 PM"


def create_slots(
    num_slots: int,
    reserve_price: float,
    start_hour: int = 9,
    slot_duration_hours: int = 1,
    num_cpus: int = 1,
) -> list[Slot]:
    """
    Create a list of consecutive time slots with uniform reserve price.
    
    Single-CPU (num_cpus=1): slot_id 0..num_slots-1, no time_index/cpu_id (backward compatible).
    Multi-CPU (num_cpus>1): slot_id = time_index + cpu_id * num_slots; each (time, cpu) is a slot.
    
    Args:
        num_slots: Number of time slots per CPU
        reserve_price: Reserve price for all slots
        start_hour: Starting hour (24-hour format)
        slot_duration_hours: Duration of each slot in hours
        num_cpus: Number of identical CPUs (1 = single-CPU, current behavior)
        
    Returns:
        List of Slot objects
    """
    slots = []
    for cpu in range(num_cpus):
        for t in range(num_slots):
            hour = start_hour + t * slot_duration_hours
            slot_id = t + cpu * num_slots
            if num_cpus == 1:
                slots.append(Slot(
                    slot_id=slot_id,
                    time_label=_time_label(hour),
                    reserve_price=reserve_price,
                ))
            else:
                slots.append(Slot(
                    slot_id=slot_id,
                    time_label=_time_label(hour),
                    reserve_price=reserve_price,
                    time_index=t,
                    cpu_id=cpu,
                ))
    return slots


def create_slots_with_prices(
    reserve_prices: list[float],
    start_hour: int = 9,
    slot_duration_hours: int = 1,
    num_cpus: int = 1,
) -> list[Slot]:
    """
    Create slots with individual reserve prices (per time step; duplicated per CPU if num_cpus > 1).
    
    Args:
        reserve_prices: Reserve price per time step (length = num_slots)
        start_hour: Starting hour (24-hour format)
        slot_duration_hours: Duration of each slot in hours
        num_cpus: Number of identical CPUs
        
    Returns:
        List of Slot objects
    """
    num_slots = len(reserve_prices)
    slots = []
    for cpu in range(num_cpus):
        for t, price in enumerate(reserve_prices):
            hour = start_hour + t * slot_duration_hours
            slot_id = t + cpu * num_slots
            if num_cpus == 1:
                slots.append(Slot(slot_id=slot_id, time_label=_time_label(hour), reserve_price=price))
            else:
                slots.append(Slot(
                    slot_id=slot_id,
                    time_label=_time_label(hour),
                    reserve_price=price,
                    time_index=t,
                    cpu_id=cpu,
                ))
    return slots
