# Sets are matched on C50 (9.00 dB both) and T30 (0.506 / 0.503 s), not just on
# room type. Splitting by type alone put the more reverberant room of four out of
# five pairs in the same set.

ROOMS_A: frozenset[str] = frozenset(
    {
        "Room9",  # bathroom1
        "Room8",  # bedroom2
        "Room2",  # living_room2
        "Room4",  # living_room_w_hallway2
        "Room5",  # meeting_room1
    }
)
ROOMS_B: frozenset[str] = frozenset(
    {
        "Room10",  # bathroom2
        "Room7",   # bedroom1
        "Room1",   # living_room1
        "Room3",   # living_room_w_hallway1
        "Room6",   # meeting_room2
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
