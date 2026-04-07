import numpy as np
from itertools import product


class LuoshuPrinciple:
    """
    Luoshu Principle:
    A recursive rule system that generates spatial structure
    via path-based decoding across scales.
    """

    def __init__(self):
        self.mapping = {
            4: (0, 0), 9: (0, 1), 2: (0, 2),
            3: (1, 0), 5: (1, 1), 7: (1, 2),
            8: (2, 0), 1: (2, 1), 6: (2, 2),
        }
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.digits = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def decode(self, path):
        """
        path: (a1, a2, ..., ak)
        return: (i, j)
        """
        i, j = 0, 0
        for a in path:
            di, dj = self.mapping[a]
            i = i * 3 + di
            j = j * 3 + dj
        return i, j

    def encode(self, i, j, k):
        """
        Reverse: (i, j) -> path of length k
        """
        path = []
        for _ in range(k):
            di = i % 3
            dj = j % 3
            path.append(self.reverse_mapping[(di, dj)])
            i //= 3
            j //= 3
        return tuple(reversed(path))

    def generate(self, k):
        """
        Generate Luoshu grid of size (3^k, 3^k)
        """
        size = 3 ** k
        grid = np.zeros((size, size), dtype=int)

        for idx, path in enumerate(product(self.digits, repeat=k), start=1):
            i, j = self.decode(path)
            grid[i, j] = idx

        return grid


if __name__ == "__main__":
    lp = LuoshuPrinciple()

    print("3x3:")
    print(lp.generate(1))
    print()

    print("9x9:")
    print(lp.generate(2))
    print()

    g27 = lp.generate(3)
    print("27x27 shape:", g27.shape)
    print("27x27 center 9x9:")
    print(g27[9:18, 9:18])
