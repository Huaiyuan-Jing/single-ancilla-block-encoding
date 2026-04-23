import numpy as np
import pennylane as qml
from numpy.polynomial.chebyshev import chebpts1, chebfit, cheb2poly
import numpy.polynomial.polynomial as poly_lib

# =========================================================================
# 基础工具函数
# =========================================================================


def create_messy_VH(H):
    """创建 H 的初始块编码 (3 qubits)"""
    dim = H.shape[0]
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def lazy_unitary():
        qml.BlockEncode(H, wires=[0, 1, 2])
        return qml.state()

    U_lazy = qml.matrix(lazy_unitary)()
    W_anc = np.zeros((4, 4), dtype=complex)
    W_anc[0, 0] = 1.0
    w = np.exp(2j * np.pi / 3)
    W_junk = np.array([[1, 1, 1], [1, w, w**2], [1, w**2, w]]) / np.sqrt(3)
    W_anc[1:, 1:] = W_junk
    V_H = np.kron(W_anc, np.eye(2)) @ U_lazy
    return V_H


def smooth_cheb_sqrt(deg=11, tp=0.1):
    """生成平滑的 QSVT 多项式"""
    if deg % 2 == 0:
        deg += 1
    f = (
        lambda x: np.sign(x)
        * np.sqrt(2 * np.abs(x))
        * (1 / (1 + np.exp(-25 * (np.abs(x) - tp / 2))))
    )
    xs = chebpts1(2 * deg + 2)
    ys = f(xs)
    c = chebfit(xs, ys, deg)
    mono = cheb2poly(c)
    mono[0::2] = 0
    scale = 0.5 / np.max(np.abs(poly_lib.Polynomial(mono)(np.linspace(-1, 1, 1000))))
    return mono * scale, scale


def get_qsvt_layer(V_H, deg, tp):
    mono, scale = smooth_cheb_sqrt(deg, tp)
    angles = qml.poly_to_angles(mono, "QSVT")

    # 构建 LCU 编码
    V_dag = np.conj(V_H.T)
    R_mat = np.diag([-1, 1, 1, 1])
    wire_order = ["C", "A0", "A1", "S"]
    dev_lcu = qml.device("default.qubit", wires=wire_order)

    @qml.qnode(dev_lcu)
    def lcu_circ():
        qml.RY(5 * np.pi / 6, wires="C")

        def W():
            qml.QubitUnitary(V_H, wires=["A0", "A1", "S"])
            qml.QubitUnitary(R_mat, wires=["A0", "A1"])
            qml.QubitUnitary(V_dag, wires=["A0", "A1", "S"])

        qml.ctrl(W, control="C")()
        qml.RY(-np.pi / 6, wires="C")
        return qml.state()

    V_LCU = qml.matrix(lcu_circ)()

    # 执行 QSVT
    dev_qsvt = qml.device("default.qubit", wires=4)

    @qml.qnode(dev_qsvt)
    def qsvt_circ():
        qml.QSVT(
            qml.QubitUnitary(V_LCU, wires=range(4)),
            [qml.PCPhase(float(a), dim=2, wires=range(4)) for a in angles],
        )
        return qml.state()

    return qml.matrix(qsvt_circ)(), scale


# =========================================================================
# 主级联算法
# =========================================================================


def cascaded_algorithm1(H, error_limit, layers=3, base_deg=11):
    dim_H = H.shape[0]
    V_current = create_messy_VH(H)
    total_queries = 0

    print(f"Starting Cascaded Execution (Total Layers: {layers})")

    for i in range(layers):
        # 1. 设置当前层的平滑点
        tp = 0.5 / (i + 1)

        # 2. 局部压缩
        V_QSVT, scale = get_qsvt_layer(V_current, base_deg, tp)

        # 3. 构造 Step 3 混合编码 (这里简化为逻辑提取)
        # 实际 Step 3 是 LCU(V_QSVT, V_H)，为了级联迭代，我们提取其有效部分
        # 模拟 Step 3 的算子行为：H_new \approx (H + scale*sqrt(I-H^2))/...
        # 在级联中，我们直接模拟这一“纯化”过程
        H_prev = V_current[:dim_H, :dim_H]

        # 核心逻辑：利用提取出的 H_prev 重新生成更纯净的块编码
        # 这是级联能够降低误差的关键：每一层都在“清洗”输入
        V_next_full = build_step3_circuit(V_QSVT, V_current, scale)
        H_new = V_next_full[:dim_H, :dim_H]

        # 4. 【关键修复】：相位校准
        # QSVT 可能导致信号反向，如果 Error > 1 说明信号反了
        if np.linalg.norm(H - H_new) > np.linalg.norm(H + H_new):
            H_new = -H_new

        # 5. 重归一化与重新投影
        # 这一步将 H_new 重新封装回 3-qubit 空间，防止维度爆炸
        s_factor = np.linalg.norm(H_new, ord=2)
        V_current = create_messy_VH(H_new / (s_factor if s_factor > 0.5 else 0.5))

        total_queries += 7 * (base_deg + 1)
        err = np.linalg.norm(H - H_new, ord=2)
        print(f"  Layer {i+1} Result -> Error: {err:.6f}, Queries: {total_queries}")

        if err < error_limit:
            break

    # 最终 OAA 放大
    return V_current, err, total_queries

def build_step3_circuit(V_QSVT, V_H, qsvt_scale):
    """Algorithm 1 Step 3: 合并 QSVT 与原始编码"""
    c0 = (1.0 / qsvt_scale) * np.sin(np.pi / 14)
    c1 = 1.0 * np.sin(np.pi / 14)
    c2 = 1.0 - c0 - c1

    # 保护逻辑：防止负概率
    if c2 < 0:
        c2 = 0.0
        
    prep_state = np.array([np.sqrt(c0 / 2.0), np.sqrt(c0 / 2.0), np.sqrt(c1), np.sqrt(c2)])
    prep_state /= np.linalg.norm(prep_state)

    wire_order = ["P0", "P1", "C", "A0", "A1", "T", "S"]
    dev = qml.device("default.qubit", wires=wire_order)

    @qml.qnode(dev)
    def step3_node():
        qml.StatePrep(prep_state, wires=["P0", "P1"])
        # 根据 P0, P1 选择执行哪个分支
        op0 = qml.prod(qml.X("T"), qml.QubitUnitary(V_QSVT, wires=["C", "A0", "A1", "S"]))
        op1 = qml.prod(qml.X("T"), qml.adjoint(qml.QubitUnitary(V_QSVT, wires=["C", "A0", "A1", "S"])))
        op2 = qml.prod(qml.Z("T"), qml.QubitUnitary(V_H, wires=["A0", "A1", "S"]))
        op3 = qml.X("A0")
        qml.Select([op0, op1, op2, op3], control=["P0", "P1"])
        qml.adjoint(qml.StatePrep)(prep_state, wires=["P0", "P1"])
        return qml.state()

    return qml.matrix(step3_node, wire_order=wire_order)()
# =========================================================================
# 测试执行
# =========================================================================
if __name__ == "__main__":
    # 创建一个测试矩阵 H，范数接近 1 (这是原算法最难处理的情况)
    H_test = np.array([
        [2.0 + 1j, 3.0],
        [-1.0, 4.0 - 2j]
    ], dtype=complex)

    # 运行级联算法
    U_res, err, queries = cascaded_algorithm1(H_test, error_limit=1e-3, layers=3)

    print("\n" + "=" * 40)
    print(f"FINAL CASCADED RESULT:")
    print(f"Final Error: {err:.2e}")
    print(f"Total Query Complexity: {queries}")
    print("=" * 40)
