from dfcc.tensor import Tensor
from dfcc.expr import Expr
from dfcc.optimizer import optimize

ERI = Tensor("V", "kilc", block="ooov")
X   = Tensor("X", "ai")
V1  = Tensor("V1", "ak")
W   = Tensor("W", "bl")
M   = Tensor("M", "cj")
Y   = Tensor("Y", "bj")

expr = Expr(ERI, X, V1, W, M, Y)
optimize(expr).emit_code()

