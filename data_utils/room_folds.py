ROOMS_A: frozenset[str] = frozenset(
    {
        "Room9",  # bathroom1
        "Room8",  # bedroom2
        "Room3",  # living_room_w_hallway1
        "Room2",  # living_room2
        "Room6",  # meeting_room2
    }
)
ROOMS_B: frozenset[str] = frozenset(
    {
        "Room10",  # bathroom2
        "Room7",  # bedroom1
        "Room4",  # living_room_w_hallway2
        "Room1",  # living_room1
        "Room5",  # meeting_room1
    }
)

ROOM_SETS: dict[str, frozenset[str]] = {
    "A": ROOMS_A,
    "B": ROOMS_B,
}

# Each value is (training rooms, validation rooms).
FOLDS: dict[int, tuple[frozenset[str], frozenset[str]]] = {
    1: (ROOMS_A, ROOMS_B),
    2: (ROOMS_B, ROOMS_A),
}
