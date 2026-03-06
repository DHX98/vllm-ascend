from pathlib import Path


def test_row_parallel_linear_registers_default_aclnn_buffers():
    source = Path("vllm_ascend/ops/linear.py").read_text()

    assert "def _ensure_aclnn_input_metadata_buffers" in source
    assert "default_device = self.weight.device" in source
    assert "self.register_buffer(buffer_name, default_value, persistent=False)" in source
    assert "self._ensure_aclnn_input_metadata_buffers()" in source


def test_sfa_o_proj_switch_materializes_and_uses_direct_aclnn_contract():
    source = Path("vllm_ascend/attention/sfa_v1.py").read_text()

    assert "self.o_proj._ensure_aclnn_input_metadata_buffers()" in source
    assert "self.o_proj_tp_aclnn_input_scale = self.o_proj.aclnn_input_scale.clone().detach()" in source


def test_sfa_skip_topk_alignment_uses_local_query_tokens_without_device_sync():
    source = Path("vllm_ascend/attention/sfa_v1.py").read_text()

    assert "num_tokens = attn_metadata.num_local_query_tokens" in source
    assert "align_topk_indices_to_actual_tokens(topk_indices, attn_metadata.num_local_query_tokens)" in source
    assert "actual_seq_lengths_query[num_seqs - 1].item()" not in source
    assert "actual_seq_lengths_query[-1].item()" not in source
    assert "query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]" in source
    assert "num_actual_seqs = int(torch.count_nonzero(query_lens_cpu))" in source
    assert "base_block_table = block_table[: li_skip_request_mask.shape[0]]" in source
    assert "torch.cat([base_block_table, base_block_table[li_skip_request_mask]], dim=0)" in source
