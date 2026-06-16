#!/usr/bin/env python3
"""Self-contained test of the subword-average init logic. Run: python test_init_logic.py"""
import numpy as np
from init_new_embeddings import average_rows


def test_mean_of_pieces():
    # old vocab of 4 tokens, dim 3
    M = np.array([[0., 0., 0.],
                  [2., 2., 2.],
                  [4., 4., 4.],
                  [6., 6., 6.]], dtype=np.float32)
    fallback = M.mean(axis=0)  # [3,3,3]
    # new token A -> pieces [1,3] => mean([2,2,2],[6,6,6]) = [4,4,4]
    # new token B -> pieces [0]   => [0,0,0]
    # new token C -> []           => fallback [3,3,3]
    rows = average_rows([[1, 3], [0], []], M, fallback)
    assert np.allclose(rows[0], [4, 4, 4]), rows[0]
    assert np.allclose(rows[1], [0, 0, 0]), rows[1]
    assert np.allclose(rows[2], [3, 3, 3]), rows[2]
    print("PASS mean_of_pieces")


def test_shapes_and_dtype():
    M = np.random.randn(100, 8).astype(np.float32)
    rows = average_rows([[1, 2, 3], [], [99]], M, M.mean(0))
    assert rows.shape == (3, 8)
    assert rows.dtype == np.float32
    print("PASS shapes_and_dtype")


def test_single_piece_identity():
    # a new token whose decomposition is exactly one old token should copy it
    M = np.random.randn(50, 16).astype(np.float32)
    rows = average_rows([[7]], M, M.mean(0))
    assert np.allclose(rows[0], M[7])
    print("PASS single_piece_identity")


if __name__ == "__main__":
    test_mean_of_pieces()
    test_shapes_and_dtype()
    test_single_piece_identity()
    print("\nAll embedding-init logic tests passed.")
