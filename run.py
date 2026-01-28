from dfcc.tensor import Tensor
from dfcc.expr import Expr
from dfcc.optimizer import optimize

def main():
    # Ask user for ERI
    eri_input = input("Enter ERI indices and block type (e.g., ackd vvov): ").split()
    eri_indices = eri_input[0]
    eri_block = eri_input[1]

    eri = Tensor("eri", eri_indices, block=eri_block)

    # Ask user for other tensors
    others_input = input("Enter other tensors (comma-separated, e.g., xai, vak, m): ")
    other_names = [name.strip() for name in others_input.split(",")]

    # Create Tensor objects (indices = same as name for simplicity)
    other_tensors = [Tensor(name, name) for name in other_names]

    # Ask for residual name (optional, default Rxy)
    res_name = input("Enter residual name (default Rxy): ").strip()
    if not res_name:
        res_name = "Rxy"

    # Build expression
    expr = Expr(eri, *other_tensors)

    # Optimize
    opt = optimize(expr, final_residual=res_name)
    opt.emit_code()


if __name__ == "__main__":
    main()

