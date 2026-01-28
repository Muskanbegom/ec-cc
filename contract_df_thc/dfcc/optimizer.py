import numpy as np
from .tensor import Tensor
from .expr import Expr

class OptimizedExpr:
    def __init__(self, final_residual="result"):
        self.code = []
        self.tmp_id = 0
        self.final_residual = final_residual

    def add(self, line):
        self.code.append(line)

    def emit_code(self):
        for line in self.code:
            print(line)


def optimize(expr: Expr, final_residual="Rxy", max_indices=3, use_df=False, use_thc=False):
    """
    Advanced optimizer with DF/THC option.
    max_indices: max allowed indices in intermediate (default 3)
    use_df: True -> use Density Fitting
    use_thc: True -> use THC
    """
    opt = OptimizedExpr(final_residual=final_residual)
    tensors = list(expr.tensors)

    def contract_pairwise(tensors):
        if len(tensors) == 1:
            return tensors[0]

        # Pick pair with shared indices
        best_pair = None
        min_len = 100
        for i in range(len(tensors)):
            for j in range(i+1, len(tensors)):
                t1, t2 = tensors[i], tensors[j]
                shared = set(t1.indices) & set(t2.indices)
                out_idx = (set(t1.indices) | set(t2.indices)) - shared
                if len(out_idx) <= max_indices and len(out_idx) < min_len:
                    best_pair = (i,j)
                    min_len = len(out_idx)

        # If no pair satisfies max_indices
        if best_pair is None:
            # Ask user to choose DF or THC
            print("\n⚠️ Contraction exceeds max indices!")
            print("Tensors:", [t.name + ":" + t.indices for t in tensors])
            print(f"Max allowed indices: {max_indices}")
            if not use_df and not use_thc:
                choice = input("Choose DF or THC? (df/thc): ").strip().lower()
                if choice == "df":
                    use_df_local = True
                    use_thc_local = False
                elif choice == "thc":
                    use_df_local = False
                    use_thc_local = True
                else:
                    raise RuntimeError("Invalid choice. Please type df or thc.")
            else:
                use_df_local = use_df
                use_thc_local = use_thc

            # Apply DF or THC factorization (placeholders)
            if use_df_local:
                t = tensors[0]  # pick first large tensor
                if len(t.indices) == 4:
                    ab = t.indices[:2]
                    cd = t.indices[2:]
                    tmp1 = f"T{opt.tmp_id}"; opt.tmp_id += 1
                    tmp2 = f"T{opt.tmp_id}"; opt.tmp_id += 1
                    opt.add(f"{tmp1} = np.einsum('P{ab},{ab}->P', L_P{ab}, X_{ab})  # DF step")
                    opt.add(f"{tmp2} = np.einsum('P{cd},P->{cd}', L_P{cd}, {tmp1})  # DF step")
                    new_tensor = Tensor(tmp2, cd)
                    tensors = [new_tensor] + tensors[1:]
                    return contract_pairwise(tensors)
            elif use_thc_local:
                t = tensors[0]  # pick first large tensor
                # placeholder THC factorization
                tmp = f"T{opt.tmp_id}"; opt.tmp_id += 1
                opt.add(f"{tmp} = THC_factorization({t.name})  # THC placeholder")
                new_tensor = Tensor(tmp, t.indices[:max_indices])
                tensors = [new_tensor] + tensors[1:]
                return contract_pairwise(tensors)
            else:
                raise RuntimeError("No contraction strategy available")

        i, j = best_pair
        t1, t2 = tensors[i], tensors[j]
        shared = set(t1.indices) & set(t2.indices)
        out_idx = "".join(sorted((set(t1.indices) | set(t2.indices)) - shared))

        tmp_name = f"T{opt.tmp_id}"
        opt.tmp_id += 1

        if len(tensors) == 2:
            opt.add(f"{final_residual} -= np.einsum('{t1.indices},{t2.indices}->{out_idx}', {t1.name}, {t2.name}, optimize='greedy')")
            new_tensor = Tensor(final_residual, out_idx)
        else:
            opt.add(f"{tmp_name} = np.einsum('{t1.indices},{t2.indices}->{out_idx}', {t1.name}, {t2.name}, optimize='greedy')")
            new_tensor = Tensor(tmp_name, out_idx)

        # build new tensor list
        new_tensors = [tensors[k] for k in range(len(tensors)) if k not in (i,j)]
        new_tensors.insert(0, new_tensor)
        return contract_pairwise(new_tensors)

    contract_pairwise(tensors)
    return opt

