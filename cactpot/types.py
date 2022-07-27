import itertools
from collections import Counter
from typing import Set, Sequence

import numpy as np

from cactpot import NULL, PAYOUT, VALID_VALS


class Vector:
    def __init__(self, v1=NULL, v2=NULL, v3=NULL, available_values: Set[int] = None):
        self.available = set(available_values) if available_values else set()
        self.known = {v1, v2, v3} - {NULL}
        if len(self.known) + len(self.available) < 3:
            raise RuntimeError('Insufficient values available')
        self._payouts: Counter = None
        self._mean: float = None
        self._variance: float = None

    @property
    def mean(self) -> float:
        """Expected value of this vector"""
        if self._mean is None:
            self._mean = float(np.mean(list(self.payouts)))
        return self._mean

    @property
    def variance(self) -> float:
        """Expected variance of this vector"""
        if self._variance is None:
            self._variance = float(np.var(list(self.payouts)))
        return self._variance

    @property
    def payouts(self) -> Counter[int]:
        """Possible payouts of this vector"""
        if self._payouts is None:
            self._payouts = Counter()
            if len(self.known) == 3:
                self._payouts.update([PAYOUT[sum(self.known)]])
            elif len(self.known) == 2:
                vals = list(self.known)
                v0, v1 = vals[0], vals[1]
                for v in self.available:
                    v_next = Vector(v0, v1, v)
                    self._payouts.update(v_next.payouts)
            elif len(self.known) == 1:
                v0 = list(self.known)[0]
                for v1, v2 in itertools.combinations(self.available, 2):
                    v_next = Vector(v0, v1, v2)
                    self._payouts.update(v_next.payouts)
            else:
                for v0, v1, v2 in itertools.combinations(self.available, 3):
                    v_next = Vector(v0, v1, v2)
                    self._payouts.update(v_next.payouts)
        return self._payouts

    def __repr__(self):
        return f'<Vector {",".join(str(i) for i in self.known)}>'

    def __str__(self):
        return f'<Vector({",".join(str(i) for i in self.known)}) μ={self.mean:,.0f} V={self.variance:,.0f}>'

    def __hash__(self):
        values = list(self.known)
        values.sort()
        values += [NULL] * (3 - len(values))
        return hash(tuple(values))


class Board:
    _markers = dict(a='→',
                    b='→',
                    c='→',
                    d='↘',
                    e='↓',
                    f='↓',
                    g='↓',
                    h='↙')

    def __init__(self, state: Sequence[int]):
        if len(state) != 9:
            raise RuntimeError(f'Invalid state {state}')
        self.state = np.array(state, dtype=np.uint8)
        self.available = VALID_VALS - set(self.state)
        self.vectors = dict(a=Vector(state[6], state[7], state[8], available_values=self.available),
                            b=Vector(state[3], state[4], state[5], available_values=self.available),
                            c=Vector(state[0], state[1], state[2], available_values=self.available),
                            d=Vector(state[0], state[4], state[8], available_values=self.available),
                            e=Vector(state[0], state[3], state[6], available_values=self.available),
                            f=Vector(state[1], state[4], state[7], available_values=self.available),
                            g=Vector(state[2], state[5], state[8], available_values=self.available),
                            h=Vector(state[2], state[4], state[6], available_values=self.available))
        self._max_direction = None
        self._total_variance = None
        self._max_tile = None

    @property
    def max_direction(self) -> str:
        """Returns vector label with maximum mean value"""
        if self._max_direction is None:
            max_label = ''
            max_v = -np.inf
            for label, vector in self.vectors.items():
                value = vector.mean
                if value > max_v:
                    max_label = label
                    max_v = value
            self._max_direction = max_label
        return self._max_direction

    @property
    def max_vector(self) -> Vector:
        """Returns vector maximum exp value"""
        return self.vectors[self.max_direction]

    def __repr__(self):
        return f'<Board {self.state}>'

    def __str__(self):
        board = [str(i) if i != NULL else '_' for i in self.state]
        board = ''.join(board)
        if self.available:
            board = board[:self.max_tile] + '*' + board[self.max_tile + 1:]
        board = 'defgh\nc' + board[0:3] + ' \nb' + board[3:6] + ' \na' + board[6:] + ' '
        board = board.replace(self.max_direction, self._markers[self.max_direction])
        for c in 'abcdefgh':
            board = board.replace(c, '.')
        return board

    @property
    def total_variance(self) -> float:
        if self._total_variance is None:
            if self.available:
                self._total_variance = np.sum(v.variance for v in self.vectors.values())
            else:
                self._total_variance = 0.0
        return self._total_variance

    def var_reduce(self, tile: int) -> float:
        """Returns average reduction in variance by revealing tile"""
        if self.state[tile] != NULL:
            raise RuntimeError(f'Tile {tile} is currently populated')
        variances = []
        for value in self.available:
            new_state = self.state.copy()
            new_state[tile] = value
            new_board = Board(new_state)
            variances.append(new_board.total_variance)
        return float(np.mean(variances) - self.total_variance)

    @property
    def max_tile(self) -> int:
        """Return tile number that reveals greatest reduction in variance"""
        if not self.available:
            raise RuntimeError(f'All tiles filled')
        if self._max_tile is not None:
            return self._max_tile
        min_reduce = np.inf
        min_tile = 9
        for tile in range(9):
            if self.state[tile] != NULL:
                continue
            reduction = self.var_reduce(tile)
            if reduction < min_reduce:
                min_reduce = reduction
                min_tile = tile
        self._max_tile = min_tile
        return min_tile
