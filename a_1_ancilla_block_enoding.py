import numpy as np
import pennylane as qml
from scipy.linalg import svd, fractional_matrix_power

# 注意：请确保 algo.py 在当前目录下
from algo import algorithm1 

def get_polar_decomposition(A):
    """
    使用 SVD 稳定地计算极分解 A = U * H
    H 是厄米特阵 |A|，U 是酉阵
    """
    u, s, vh = svd(A)
    U = u @ vh
    H = vh.T.conj() @ np.diag(s) @ vh
    return U, H

def general_matrix_block_encoding(A, error_limit=1e-3):
    """
    实现一般矩阵 A 的单比特（基于 algo1）Block Encoding
    """
    # 1. 预归一化
    norm_A = np.linalg.norm(A, ord=2)
    # Block Encoding 要求算子范数 <= 1
    A_scaled = A / norm_A if norm_A > 1.0 else A
    
    # 2. 经典计算极分解 (这是为了避开 QSVT 角度求解失败的坑)
    print("-> 正在进行经典极分解...")
    U_matrix, H_matrix = get_polar_decomposition(A_scaled)
    
    # 3. 使用 algo1 构造 H = |A| 的 Block Encoding
    print("-> 正在利用 algo1 构造 Hermitian 部分 |A| 的 Block Encoding...")
    # algorithm1 会返回一个大的酉矩阵 V_H，其左上角是 H
    V_H, err_H, queries = algorithm1(H_matrix, error_limit)
    
    # 4. 构造系统比特上的酉变换 U
    # 假设 algo1 的系统比特在最后一维（根据 algo.py 逻辑）
    total_dim = V_H.shape[0]
    sys_dim = A.shape[0]
    ancilla_dim = total_dim // sys_dim
    
    # 在全空间上构造 U_full = I(ancilla) ⊗ U(system)
    # 这相当于在量子线路上只对系统比特做 U 变换，辅助比特不动
    U_full = np.kron(np.eye(ancilla_dim), U_matrix)
    
    # 5. 组合：V_A = U_full @ V_H
    # 物理意义：先跑 algo1 线路，再在系统比特上挂一个 U 门
    V_A = U_full @ V_H
    
    # 6. 验证结果
    A_extracted = V_A[:sys_dim, :sys_dim]
    # 别忘了还原缩放倍数
    if norm_A > 1.0:
        A_extracted *= norm_A
        
    final_err = np.linalg.norm(A - A_extracted, ord=2)
    
    return V_A, final_err

# ==========================================
# 测试运行
# ==========================================
if __name__ == "__main__":
    # 定义一个非厄米特的矩阵
    A = np.array([[0.1, 0.4], 
                  [-0.2, 0.3]])
    
    try:
        V_final, err = general_matrix_block_encoding(A, error_limit=1e-2)
        print(f"\n[成功] 最终 Block Encoding 误差: {err:.2e}")
        print(f"矩阵维度: {V_final.shape[0]} (包含辅助比特)")
        
        # 打印左上角检查
        print("\n提取的矩阵 A_approx:")
        print(np.real(V_final[:2, :2]))
        print("\n", np.real(V_final))
    except Exception as e:
        print(f"\n[错误] 运行失败: {e}")