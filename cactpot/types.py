import itertools
from typing import Set, Sequence

import numpy as np

from cactpot import NULL, PAYOUT, VALID_VALS


class Vector:
    def __init__(self, v1=NULL, v2=NULL, v3=NULL, available_values: Set[int] = None):
        self.available = set(available_values) if available_values else set()
        self.known = {v1, v2, v3} - {NULL}
        if len(self.known) + len(self.available) < 3:
            raise RuntimeError('Insufficient values available')
        self._mean: float = None
        self._variance: float = None

    @property
    def mean(self) -> float:
        """Expected value of this vector"""
        if self._mean is not None:
            return self._mean
        self.populate_stats()
        return self._mean

    @property
    def variance(self) -> float:
        """Expected variance of this vector"""
        if self._variance is not None:
            return self._variance
        self.populate_stats()
        return self._variance

    def populate_stats(self):
        if len(self.known) == 3:
            self._variance = 0
            self._mean = PAYOUT[sum(self.known)]
        else:
            sum_known = sum(self.known) if self.known else 0
            pick = 3 - len(self.known)
            payouts = []
            for selection in itertools.combinations(self.available, pick):
                payouts.append(PAYOUT[sum_known + sum(selection)])
            self._mean = float(np.mean(payouts))
            self._variance = float(np.var(payouts))

    def __repr__(self):
        return f'<Vector {",".join(str(i) for i in self.known)}>'

    def __str__(self):
        return f'<Vector({",".join(str(i) for i in self.known)}) μ={self.mean:,.0f} V={self.variance:,.0f}>'


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
        self._max_vector = None
        self._total_variance = None
        self._max_tile = None

    @property
    def max_vector(self) -> str:
        if self._max_vector is None:
            max_k = ''
            max_v = -np.inf
            for k, v in self.vectors.items():
                value = v.mean
                if value > max_v:
                    max_k = k
                    max_v = value
            self._max_vector = max_k
        return self._max_vector

    def __repr__(self):
        return f'<Board {self.state}>'

    def __str__(self):
        board = [str(i) if i != NULL else '_' for i in self.state]
        board = ''.join(board)
        if self.available:
            board = board[:self.max_tile] + '*' + board[self.max_tile + 1:]
        board = 'defgh\nc' + board[0:3] + ' \nb' + board[3:6] + ' \na' + board[6:] + ' '
        board = board.replace(self.max_vector, self._markers[self.max_vector])
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

