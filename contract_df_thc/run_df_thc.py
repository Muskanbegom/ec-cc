from dfcc.tensor import Tensor
from dfcc.expr import Expr
from dfcc.optimizer import optimize

def main():
    # ERI input
    eri_input = input("Enter ERI indices and block type (e.g., ackd vvov): ").split()
    eri_indices = eri_input[0]
    eri_block = eri_input[1]
    eri = Tensor("eri", eri_indices, block=eri_block)

    # Other tensors
    others_input = input("Enter other tensors (comma-separated, e.g., xai, vak, m): ")
    other_names = [name.strip() for name in others_input.split(",")]
    other_tensors = [Tensor(name, name) for name in other_names]

    # Residual
    res_name = input("Enter residual name (default Rxy): ").strip() or "Rxy"

    # Build expr
    expr = Expr(eri, *other_tensors)

    # Optimize with max_indices=3
    opt = optimize(expr, final_residual=res_name, max_indices=3)
    opt.emit_code()

if __name__ == "__main__":
    main()

