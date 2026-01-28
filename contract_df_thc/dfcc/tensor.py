class Tensor:
    def __init__(self, name, indices, block=None, df=False):
        self.name = name
        self.indices = indices      # string, e.g. "abcd"
        self.block = block          # e.g. "vvvv"
        self.df = df                # DF-allowed?

    def rank(self):
        return len(self.indices)

    def __repr__(self):
        return f"{self.name}_{self.indices}"

