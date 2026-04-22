import numpy as np
import pennylane as qml


def create_messy_VH(H):
    """
    Creates a mathematically perfect but chaotic block encoding of H.
    Assumes H is already Hermitian and its norm is <= 1.0.
    """
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def lazy_unitary():
        # BlockEncode automatically requires ||H|| <= 1, which our main.py now guarantees
        qml.BlockEncode(H, wires=[0, 1, 2])
        return qml.state()

    U_lazy = qml.matrix(lazy_unitary)()

    W_anc = np.zeros((4, 4), dtype=complex)
    W_anc[0, 0] = 1.0
    w = np.exp(2j * np.pi / 3)
    W_junk = np.array([[1, 1, 1], [1, w, w**2], [1, w**2, w]]) / np.sqrt(3)
    W_anc[1:, 1:] = W_junk

    V_H = np.kron(W_anc, np.eye(2)) @ U_lazy

    # We no longer return a separately normalized H_norm, just the H we were given
    return H, V_H


def build_lcu_encoding(V_matrix):
    """Builds LCU Encoding of (I-H^2)/2"""
    wire_order = ["C", "A_0", "A_1", "S"]
    dev = qml.device("default.qubit", wires=wire_order)

    V_dagger = np.conj(V_matrix.T)
    R_mat = np.diag([-1, 1, 1, 1])

    @qml.qnode(dev)
    def circ(V_mat, V_dag):
        qml.RY(5 * np.pi / 6, wires="C")

        def apply_W():
            qml.QubitUnitary(V_mat, wires=["A_0", "A_1", "S"])
            qml.QubitUnitary(R_mat, wires=["A_0", "A_1"])
            qml.QubitUnitary(V_dag, wires=["A_0", "A_1", "S"])

        qml.ctrl(apply_W, control="C")()
        qml.RY(-np.pi / 6, wires="C")
        return qml.state()

    U_LCU = qml.matrix(circ, wire_order=wire_order)(V_matrix, V_dagger)
    return U_LCU, U_LCU[:2, :2]


def cheb_sqrt2x(deg=21, max_scale=0.5):
    """Chebyshev polynomial approximation for sqrt(2x)"""
    if deg % 2 == 0:
        deg += 1
    f = lambda x: np.sign(x) * np.sqrt(2 * np.abs(x))
    xs = np.polynomial.chebyshev.chebpts1(2 * deg)
    ys = f(xs)
    c = np.polynomial.chebyshev.chebfit(xs, ys, deg)
    mono = np.polynomial.chebyshev.cheb2poly(c)
    mono[0::2] = 0
    poly = np.polynomial.polynomial.Polynomial(mono)
    grid = np.linspace(-1, 1, 4000)
    scale = max_scale / np.max(np.abs(poly(grid)))
    return mono * scale, scale


def qsvt_on_lcu(V_LCU, deg=21):
    """Applies QSVT using the polynomial calculated"""
    mono, scale = cheb_sqrt2x(deg)
    angles = qml.poly_to_angles(mono, "QSVT")

    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circ():
        U = qml.QubitUnitary(V_LCU, wires=range(4))
        P = [qml.PCPhase(float(a), dim=2, wires=range(4)) for a in angles]
        qml.QSVT(U, P)
        return qml.state()

    Q = qml.matrix(circ)()
    return Q, scale


def get_qsvt_and_vh(H, deg):
    """High-level wrapper to produce V_QSVT and V_H directly from target H"""
    H_norm, V_H = create_messy_VH(H)
    V_LCU, _ = build_lcu_encoding(V_H)
    V_QSVT, scale = qsvt_on_lcu(V_LCU, deg=deg)
    return V_QSVT, V_H, scale, H_norm


def oaa_reflection(U, good_dim):
    """Implements the reflection oracle for Oblivious Amplitude Amplification."""
    dim = U.shape[0]
    R = np.eye(dim, dtype=complex)
    for i in range(good_dim):
        R[i, i] = -1.0
    return -U @ R @ U.conj().T @ R


def build_step3_circuit(V_QSVT, V_H, qsvt_scale):
    """Combines QSVT matrix and original block encoding into the Walk Operator."""
    c0 = (1.0 / qsvt_scale) * np.sin(np.pi / 14)
    c1 = 1.0 * np.sin(np.pi / 14)
    c2 = 1.0 - c0 - c1

    # FIX: Check if the scale is physically valid for this probability distribution
    if c2 < 0:
        raise ValueError(
            f"QSVT scale ({qsvt_scale:.3f}) is too small. LCU weights exceed 1.0."
        )

    prep_state = np.array(
        [np.sqrt(c0 / 2.0), np.sqrt(c0 / 2.0), np.sqrt(c1), np.sqrt(c2)]
    )

    wire_order = ["P0", "P1", "C", "A0", "A1", "T", "S"]
    dev = qml.device("default.qubit", wires=wire_order)

    @qml.qnode(dev)
    def step3_node():
        qml.StatePrep(prep_state, wires=["P0", "P1"])

        op0 = qml.prod(
            qml.X("T"), qml.QubitUnitary(V_QSVT, wires=["C", "A0", "A1", "S"])
        )
        op1 = qml.prod(
            qml.X("T"),
            qml.adjoint(qml.QubitUnitary(V_QSVT, wires=["C", "A0", "A1", "S"])),
        )
        op2 = qml.prod(qml.Z("T"), qml.QubitUnitary(V_H, wires=["A0", "A1", "S"]))
        op3 = qml.X("A0")

        qml.Select([op0, op1, op2, op3], control=["P0", "P1"])
        qml.adjoint(qml.StatePrep)(prep_state, wires=["P0", "P1"])

        return qml.state()

    return qml.matrix(step3_node, wire_order=wire_order)()


# =========================================================================
# MAIN WRAPPER: Algorithm 1
# =========================================================================


def algorithm1(H, error_limit, max_deg=1001):
    """
    Implements Algorithm 1: Approximate Hermitian Block Encoding with One Ancilla.
    """
    dim_H = H.shape[0]
    good_dim = dim_H * 2
    rounds = 3

    deg = 5
    best_U = None
    best_error = float("inf")
    best_queries = 0

    while deg <= max_deg:

        try:
            # 1. Generate building blocks (MUST BE INSIDE TRY BLOCK)
            V_QSVT, V_H, qsvt_scale, H_norm = get_qsvt_and_vh(H, deg)

            # 2. Build the Step 3 LCU un-amplified walk operator
            U_step3 = build_step3_circuit(V_QSVT, V_H, qsvt_scale)

        except ValueError as e:
            # Check if this is the QSVT classical angle solver breaking down
            if "inhomogeneous" in str(e) or "sequence" in str(e):
                print(
                    f"  -> Notice: Classical QSVT angle solver reached numerical instability limit at deg {deg}."
                )
                break  # Stop searching, higher degrees will mathematically fail
            else:
                # Otherwise, it's the LCU probability scale issue (c2 < 0)
                deg += 4
                continue
        except Exception as e:
            print(f"  -> Notice: Unknown solver error at deg {deg} - {e}")
            break

        # 3. Step 4: Oblivious Amplitude Amplification
        Q = oaa_reflection(U_step3, good_dim)
        U_current = U_step3.copy()

        for _ in range(rounds):
            U_current = Q @ U_current

        final_block_encoding = U_current

        # 4. Validate Error
        H_extracted = final_block_encoding[:dim_H, :dim_H]
        error = np.linalg.norm(H_norm - H_extracted, ord=2)

        # 5. Query Count Calculation
        queries_per_step3 = deg + 1
        total_queries = 7 * queries_per_step3

        # Save the best achievable valid result
        best_error = error
        best_U = final_block_encoding
        best_queries = total_queries

        if error < error_limit:
            return final_block_encoding, error, total_queries

        deg += 4

    return best_U, best_error, best_queries
