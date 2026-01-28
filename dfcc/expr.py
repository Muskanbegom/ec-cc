class Expr:
    def __init__(self, *tensors):
        self.tensors = list(tensors)

    def all_indices(self):
        idx = ""
        for t in self.tensors:
            idx += t.indices
        return set(idx)

    def __repr__(self):
        return " * ".join(map(str, self.tensors))

