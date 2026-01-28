import numpy as np
from .tensor import Tensor
from .expr import Expr

class OptimizedExpr:
    def __init__(self, final_residual=None):
        self.code = []
        self.tmp_id = 0
        self.final_residual = final_residual  # e.g., "Rxy"

    def add(self, line):
        self.code.append(line)

    def extend(self, other):
        self.code.extend(other.code)
        self.tmp_id = other.tmp_id

    def emit_code(self):
        for line in self.code:
            print(line)


def optimize(expr: Expr, final_residual="result"):
    opt = OptimizedExpr(final_residual=final_residual)
    tensors = list(expr.tensors)

    def contract_pairwise(tensors):
        if len(tensors) == 1:
            return tensors[0], opt

        # 1️⃣ Check for DF vvvv tensor first
        for i, t in enumerate(tensors):
            if t.block == "vvvv" and t.df:
                ab = t.indices[:2]
                cd = t.indices[2:]
                tmp_name = f"T{opt.tmp_id}"
                opt.tmp_id += 1
                opt.add(f"{tmp_name} = np.einsum('P{ab},{ab}->P', L_P{ab}, X_{ab})")
                tmp_name2 = f"T{opt.tmp_id}"
                opt.tmp_id += 1
                opt.add(f"{tmp_name2} = np.einsum('P{cd},P->{cd}', L_P{cd}, {tmp_name})")
                new_tensor = Tensor(tmp_name2, cd)
                new_list = tensors[:i] + [new_tensor] + tensors[i+1:]
                return contract_pairwise(new_list)

        # 2️⃣ Pick best pair to contract (≤4 output indices preferred)
        best_pair = None
        min_len = 100
        for i in range(len(tensors)):
            for j in range(i+1, len(tensors)):
                t1, t2 = tensors[i], tensors[j]
                shared = set(t1.indices) & set(t2.indices)
                out_idx = (set(t1.indices) | set(t2.indices)) - shared
                if len(out_idx) <= 4 and len(out_idx) < min_len:
                    best_pair = (i, j)
                    min_len = len(out_idx)

        if best_pair is None:
            i, j = 0, 1
        else:
            i, j = best_pair

        t1, t2 = tensors[i], tensors[j]
        shared = set(t1.indices) & set(t2.indices)
        out_idx = "".join(sorted((set(t1.indices) | set(t2.indices)) - shared))
        tmp_name = f"T{opt.tmp_id}"
        opt.tmp_id += 1

        # For the final contraction, assign to residual
        if len(tensors) == 2:
            opt.add(f"{opt.final_residual} -= np.einsum('{t1.indices},{t2.indices}->{out_idx}', {t1.name}, {t2.name}, optimize='greedy')")
            new_tensor = Tensor(opt.final_residual, out_idx)
        else:
            opt.add(f"{tmp_name} = np.einsum('{t1.indices},{t2.indices}->{out_idx}', {t1.name}, {t2.name}, optimize='greedy')")
            new_tensor = Tensor(tmp_name, out_idx)

        # Build new tensor list
        new_tensors = [tensors[k] for k in range(len(tensors)) if k not in (i,j)]
        new_tensors.insert(0, new_tensor)
        return contract_pairwise(new_tensors)

    contract_pairwise(tensors)
    return opt

