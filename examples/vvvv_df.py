import numpy as np
from dfcc.tensor import Tensor
from dfcc.expr import Expr
from dfcc.optimizer import optimize

V = Tensor("V", "abcd", block="vvvv", df=True)
X = Tensor("X", "ab")

expr = Expr(V, X)
opt = optimize(expr)
opt.emit_code()

