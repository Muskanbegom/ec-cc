class Index:
    def __init__(self, name, space):
        assert space in ("o", "v", "P")
        self.name = name
        self.space = space

    def __repr__(self):
        return self.name

