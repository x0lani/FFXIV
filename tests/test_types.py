from collections import Counter

import pytest

from cactpot import NULL, PAYOUT
from cactpot.types import Vector, Board


class TestVector:
    def test_value(self):
        v = Vector(4, 5, 6)
        assert v.mean == PAYOUT[15], "Failed all values filled test"
        v = Vector(1, 2, NULL, {3})
        assert v.mean == PAYOUT[6]
        v = Vector(1, 2, NULL, {3, 4})
        assert v.mean == (PAYOUT[6] + PAYOUT[7]) / 2
        v = Vector(1, 2, NULL, {3, 4, 5})
        assert v.mean == (PAYOUT[6] + PAYOUT[7] + PAYOUT[8]) / 3
        v = Vector(1, NULL, NULL, {2, 3})
        assert v.mean == PAYOUT[6]
        v = Vector(1, NULL, NULL, {2, 3, 5})
        assert v.mean == (PAYOUT[6] + PAYOUT[8] + PAYOUT[9]) / 3
        v = Vector(NULL, NULL, 1, {2, 3})
        assert v.mean == PAYOUT[6], "Failed order test"
        v = Vector(NULL, NULL, 1, {2, 3, 5})
        assert v.mean == (PAYOUT[6] + PAYOUT[8] + PAYOUT[9]) / 3, "Failed order test"
        v = Vector(NULL, NULL, NULL, {2, 3, 5})
        assert v.mean == PAYOUT[10], "Failed all NULL"
        v = Vector(NULL, NULL, NULL, {3, 4, 5, 6})
        assert v.mean == (PAYOUT[12] + PAYOUT[13] + PAYOUT[14] + PAYOUT[15]) / 4, "Failed all NULL"
        v = Vector(available_values={3, 4, 5, 6})
        assert v.mean == (PAYOUT[12] + PAYOUT[13] + PAYOUT[14] + PAYOUT[15]) / 4, "Failed all NULL"
        with pytest.raises(RuntimeError):
            v = Vector(1, NULL, NULL, {2}), "Not enough available values to fill NULLs"
            print(v.value == 0)
        with pytest.raises(RuntimeError):
            v = Vector(1, NULL, NULL), "Not enough available values to fill NULLs"
            print(v.value == 0)
        with pytest.raises(RuntimeError):
            v = Vector(NULL, NULL, NULL, {2, 3}), "Not enough available values to fill NULLs"
            print(v.value == 0)

    def test_variance(self):
        v = Vector(1, 2, 3)
        assert v.variance == 0, "Failed all values filled"
        v = Vector(1, 2, 3, {4, 5, 6})
        assert v.variance == 0, "Failed all values filled"
        v = Vector(1, 2, NULL, {3})
        assert v.variance == 0
        v = Vector(1, 2, NULL, {3, 4})
        assert v.variance == 24820324
        v = Vector(1, 2, NULL, {3, 4, 5, 6})
        assert v.variance == 17439483
        v = Vector(1, NULL, NULL, {3, 4})
        assert v.variance == 0
        v = Vector(1, NULL, NULL, {2, 3, 4})
        assert v.variance == 20651950.22222222
        v = Vector(NULL, NULL, NULL, {2, 3, 4})
        assert v.variance == 0

    def test_payouts(self):
        v = Vector(1, 2, 3)
        assert v.payouts == Counter([PAYOUT[1 + 2 + 3]])
        v = Vector(4, 5, 6)
        assert v.payouts == Counter([PAYOUT[4 + 5 + 6]])
        v = Vector(1, 2, NULL, {3})
        assert v.payouts == Counter([PAYOUT[1 + 2 + 3]])
        v = Vector(1, 2, NULL, {3, 4, 5})
        assert v.payouts == Counter([PAYOUT[6], PAYOUT[7], PAYOUT[8]])
        v = Vector(1, NULL, NULL, {2, 3})
        assert v.payouts == Counter([PAYOUT[6]])
        v = Vector(1, NULL, NULL, {2, 3, 4})
        assert v.payouts == Counter([PAYOUT[6], PAYOUT[7], PAYOUT[8]])
        v = Vector(NULL, NULL, NULL, {2, 3, 4})
        assert v.payouts == Counter([PAYOUT[9]])

    def test_hash(self):
        v1 = Vector(1, 2, 3)
        v2 = Vector(3, 2, 1)
        assert hash(v1) == hash(v2)
        v3 = Vector(1, 2, 4)
        assert hash(v1) != hash(v3)
        v1 = Vector(1, 2, NULL, available_values={7})
        v2 = Vector(NULL, 2, 1, available_values={7})
        v3 = Vector(3, 2, NULL, available_values={7})
        v4 = Vector(1, 2, 3)
        assert hash(v1) == hash(v2)
        assert hash(v1) != hash(v3)
        assert hash(v1) != hash(v4)
        v1 = Vector(5, NULL, NULL, available_values={7, 8})
        v2 = Vector(NULL, NULL, 5, available_values={7, 8})
        assert hash(v1) == hash(v2)
        v1 = Vector(NULL, NULL, NULL, available_values={7, 8, 9})
        v2 = Vector(NULL, NULL, NULL, available_values={7, 8, 9})
        assert hash(v1) == hash(v2)


class TestBoard:
    def test_board(self):
        with pytest.raises(RuntimeError):
            b = Board([1, 2, 3, 4]), "Insufficient values"
        with pytest.raises(RuntimeError):
            b = Board(list(range(20))), "too many values"
        b = Board([1, 2, 3, 4, 5, NULL, NULL, NULL, NULL])
        assert b.available == {6, 7, 8, 9}
        b = Board([NULL, NULL, NULL, NULL, 1, 2, 3, 4, 5])
        assert b.available == {6, 7, 8, 9}
        b = Board([1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert b.available == set()
        assert b.vectors['a'].mean == PAYOUT[24]
        assert b.vectors['b'].mean == PAYOUT[15]
        assert b.vectors['c'].mean == PAYOUT[6]
        assert b.vectors['d'].mean == PAYOUT[15]
        assert b.vectors['e'].mean == PAYOUT[12]
        assert b.vectors['f'].mean == PAYOUT[15]
        assert b.vectors['g'].mean == PAYOUT[18]
        assert b.vectors['h'].mean == PAYOUT[15]
        assert b.max_direction == 'c'
        b = Board([NULL, 2, 3,
                   4, NULL, 6,
                   7, 8, NULL])
        assert b.available == {1, 5, 9}
        assert b.vectors['a'].mean == Vector(7, 8, NULL, {1, 5, 9}).mean
        assert b.vectors['d'].mean == Vector(available_values={1, 5, 9}).mean
        assert b.vectors['f'].mean == Vector(2, NULL, 8, {1, 5, 9}).mean
