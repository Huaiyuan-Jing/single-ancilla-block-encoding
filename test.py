import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Import the functions from your script (assuming it's named algo.py)
# Replace 'algo' with the actual name of your python file if different.
from algo import algorithm1, oaa_reflection


# ==========================================
# TEST 1: OAA Reflection Algebra
# ==========================================
def test_oaa_reflection():
    """
    Tests the Oblivious Amplitude Amplification reflection matrix logic.
    The reflection should flip the sign of the good subspace.
    """
    # 4x4 Identity unitary for testing
    U_dummy = np.eye(4, dtype=complex)
    good_dim = 2

    Q = oaa_reflection(U_dummy, good_dim)

    # R is a matrix with -1 on the first 'good_dim' diagonal entries, 1 elsewhere
    expected_R = np.diag([-1.0, -1.0, 1.0, 1.0])

    # Since U is Identity, Q = I @ R @ I^dagger @ R = R @ R = Identity
    # However, because of how Q is built: U @ R @ U_dagger @ R
    expected_Q = U_dummy @ expected_R @ U_dummy.conj().T @ expected_R

    np.testing.assert_array_almost_equal(Q, expected_Q)
    assert Q.shape == (4, 4), "Q matrix dimension should match U"


# ==========================================
# TEST 2: Algorithm 1 - Convergence & Queries
# ==========================================
@patch("algo.build_step3_circuit")
def test_algorithm1_convergence(mock_build_step3):
    """
    Tests if algorithm1 correctly terminates when the error limit is reached
    and accurately calculates the query count[cite: 594].
    """
    dim_H = 2
    # Create a dummy 2x2 Hermitian target matrix H (sub-normalized so ||H|| <= 1)
    H_target = np.array([[0.5, 0.1], [0.1, 0.5]])

    error_limit = 0.05
    good_dim = dim_H * 2  # 4

    # --- Mocking the Quantum Outputs ---
    # We will simulate the loop running twice.
    # Iteration 1 (deg=5): High error, fails the limit check.
    # Iteration 2 (deg=9): Low error, passes the limit check.

    # Bad unitary where top-left 2x2 is far from H_target
    bad_unitary = np.eye(good_dim * 2, dtype=complex)
    bad_unitary[:dim_H, :dim_H] = np.array([[0.9, 0.9], [0.9, 0.9]])

    # Good unitary where top-left 2x2 perfectly matches H_target
    good_unitary = np.eye(good_dim * 2, dtype=complex)
    good_unitary[:dim_H, :dim_H] = H_target

    # We mock build_step3_circuit to return these matrices in sequence
    mock_build_step3.side_effect = [bad_unitary, good_unitary]

    # Patch the internal get_qsvt_and_vh call if you added it to algo.py
    with patch("algo.get_qsvt_and_vh", return_value=(MagicMock(), MagicMock(), 0.5)):

        final_U, error, queries = algorithm1(H_target, error_limit, max_deg=20)

        # --- Assertions ---
        # 1. It should have run the loop exactly twice (deg=5, then deg=9)
        assert mock_build_step3.call_count == 2

        # 2. Error should be virtually 0 (since good_unitary matches H_target exactly)
        assert error < error_limit
        np.testing.assert_almost_equal(error, 0.0)

        # 3. Verify Query Calculation for the successful degree (deg = 9)
        # Query formula: 7 * (deg + 1)
        expected_queries = 7 * (9 + 1)
        assert (
            queries == expected_queries
        ), f"Expected {expected_queries} queries, got {queries}"


# ==========================================
# TEST 3: Algorithm 1 - Max Degree Cutoff
# ==========================================
@patch("algo.build_step3_circuit")
def test_algorithm1_max_degree_cutoff(mock_build_step3):
    """
    Tests that algorithm1 stops and returns the best effort if max_deg is exceeded
    without hitting the error_limit constraint.
    """
    dim_H = 2
    H_target = np.array([[0.5, 0.0], [0.0, 0.5]])

    error_limit = 0.0001  # impossibly strict for the mock
    max_deg = 10  # Will test deg=5, deg=9, then exit

    # Mock always returns a bad approximation
    bad_unitary = np.eye(dim_H * 4, dtype=complex)
    bad_unitary[:dim_H, :dim_H] = np.array([[0.9, 0.9], [0.9, 0.9]])
    mock_build_step3.return_value = bad_unitary

    with patch("algo.get_qsvt_and_vh", return_value=(MagicMock(), MagicMock(), 0.5)):

        final_U, error, queries = algorithm1(H_target, error_limit, max_deg=max_deg)

        # --- Assertions ---
        # 1. Ensure loop ran for deg=5 and deg=9, and stopped
        assert mock_build_step3.call_count == 2

        # 2. Error should still be high (above error_limit)
        assert error > error_limit

        # 3. Queries should reflect the last valid iteration checked (deg=9)
        expected_queries = 7 * (9 + 1)
        assert queries == expected_queries
