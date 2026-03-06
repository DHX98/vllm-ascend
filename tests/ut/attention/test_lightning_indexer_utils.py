import numpy as np
import torch

from vllm_ascend.attention.utils import (
    align_topk_indices_to_actual_tokens,
    get_index_of_skipped_queries_numpy,
    get_sfa_skip_indices,
    hidden_states_reorder,
    maybe_pad_and_reorder_inputs,
)


def test_get_sfa_skip_indices_with_threshold():
    indices, li_cum_query_lens, li_seq_lens, li_skipped_mask = get_sfa_skip_indices(
        num_computed_tokens=np.array([0, 2050], dtype=np.int32),
        query_lens=np.array([10, 10], dtype=np.int32),
        skip_threshold=2048,
    )

    assert np.array_equal(indices, np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(li_cum_query_lens, np.array([0, 10, 20], dtype=np.int64))
    assert np.array_equal(li_seq_lens, np.array([10, 2060, 10], dtype=np.int64))
    assert np.array_equal(li_skipped_mask, np.array([True, False]))


def test_get_sfa_skip_indices_no_skip():
    result = get_sfa_skip_indices(
        num_computed_tokens=np.array([4096], dtype=np.int32),
        query_lens=np.array([16], dtype=np.int32),
        skip_threshold=2048,
    )
    assert result == (None, None, None, None)


def test_get_index_of_skipped_queries_numpy():
    topk = get_index_of_skipped_queries_numpy(
        actual_seq_lengths_query=np.array([2, 4, 5], dtype=np.int32),
        actual_seq_lengths_key=np.array([8, 8, 2], dtype=np.int32),
        num_actual_seqs=2,
        sparse_count=6,
    )
    assert topk.shape == (1, 1, 6)
    assert np.array_equal(topk[0, 0], np.array([0, 1, -1, -1, -1, -1], dtype=np.int32))


def test_align_topk_indices_to_actual_tokens_pad_and_trim():
    topk = torch.tensor([[[0, 1, -1]]], dtype=torch.int32)

    padded = align_topk_indices_to_actual_tokens(topk, 3)
    assert padded.shape == (3, 1, 3)
    assert torch.equal(padded[0], topk[0])
    assert torch.equal(padded[1], torch.full((1, 3), -1, dtype=torch.int32))
    assert torch.equal(padded[2], torch.full((1, 3), -1, dtype=torch.int32))

    oversized = torch.tensor(
        [
            [[0, 1, -1]],
            [[0, 1, 2]],
            [[0, 1, 2]],
            [[0, 1, 2]],
        ],
        dtype=torch.int32,
    )
    trimmed = align_topk_indices_to_actual_tokens(oversized, 2)
    assert trimmed.shape == (2, 1, 3)
    assert torch.equal(trimmed, oversized[:2])


def test_maybe_pad_and_reorder_inputs_and_restore_hidden_states():
    input_ids = torch.tensor([1, 2, 3, 0, 0], dtype=torch.int32)
    positions = torch.tensor([0, 1, 2, 0, 0], dtype=torch.int64)
    reorder_indices = torch.tensor([2, 0, 1], dtype=torch.int32)

    reordered_input_ids, reordered_positions = maybe_pad_and_reorder_inputs(input_ids, positions, reorder_indices)
    assert torch.equal(reordered_input_ids, torch.tensor([3, 1, 2, 0, 0], dtype=torch.int32))
    assert torch.equal(reordered_positions, torch.tensor([2, 0, 1, 0, 0], dtype=torch.int64))

    reordered_hidden_states = torch.tensor([[30.0], [10.0], [20.0]])
    restored_hidden_states = hidden_states_reorder(reordered_hidden_states, reorder_indices)
    assert torch.equal(restored_hidden_states, torch.tensor([[10.0], [20.0], [30.0]]))
