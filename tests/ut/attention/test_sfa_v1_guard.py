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


def test_sfa_skip_topk_alignment_uses_local_query_slots_without_device_sync():
    source = Path("vllm_ascend/attention/sfa_v1.py").read_text()

    assert "max_num_option2_reqs = max_num_reqs * 2" in source
    assert "num_tokens = attn_metadata.num_local_indexer_tokens" in source
    assert "num_local_indexer_tokens=num_local_indexer_tokens" in source
    assert "num_local_query_slots=num_local_query_slots" in source
    assert "num_local_query_slots = num_input_tokens" in source
    assert "num_local_query_slots = num_tokens_per_device" in source
    assert "align_topk_indices_to_query_slots(topk_indices, attn_metadata.num_local_query_slots)" in source
    assert "actual_seq_lengths_query[num_seqs - 1].item()" not in source
    assert "actual_seq_lengths_query[-1].item()" not in source
    assert "query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]" in source
    assert "num_actual_seqs = lightning_indexer_metadata.num_actual_reqs" in source
    assert "cum_query_lens_cpu_for_dsa_cp = lightning_indexer_metadata.li_cum_query_lens_cpu" in source
    assert "seq_lens_cpu_for_dsa_cp = lightning_indexer_metadata.li_seq_lens_cpu" in source
    assert "block_table = common_attn_metadata.block_table_tensor[:num_actual_seqs]" in source
    assert "torch.cat([block_table, block_table[li_skip_request_mask]], dim=0)" in source
    assert "num_segs = num_total_seqs" in source
    assert "raw_num_actual_reqs = getattr(common_attn_metadata, \"num_actual_reqs\", 0)" in source
    assert "block_table = common_attn_metadata.block_table_tensor[:num_actual_reqs]" in source
    assert source.index("input_positions = input_positions_pad") < source.index("cos, sin = get_cos_and_sin_mla(input_positions, True)")


def test_model_runner_tracks_actual_request_count_separately_from_padding():
    source = Path("vllm_ascend/worker/model_runner_v1.py").read_text()

    assert "num_actual_reqs = int(np.count_nonzero(num_scheduled_tokens_np))" in source
    assert "actual_num_scheduled_tokens_np = num_scheduled_tokens_np[:num_actual_reqs]" in source
    assert "li_restore_indices = np.argsort(li_reorder_indices, kind=\"stable\").astype(np.int32)" in source
    assert "li_restore_indices=torch.from_numpy(li_restore_indices)" in source
    assert "hidden_states_reorder(hidden_states, self.lightning_indexer_metadata.li_restore_indices)" in source
    assert "actual_num_scheduled_tokens = num_scheduled_tokens.copy()" in source
    assert "num_reqs_padded=num_reqs_padded" in source
    assert "num_actual_reqs=num_actual_reqs" in source
