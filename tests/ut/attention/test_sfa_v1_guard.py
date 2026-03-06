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


def test_sfa_skip_topk_alignment_uses_cum_query_lens_tail():
    source = Path("vllm_ascend/attention/sfa_v1.py").read_text()

    assert "actual_query_tokens = int(actual_seq_lengths_query[-1].item())" in source
    assert "align_topk_indices_to_actual_tokens(topk_indices, actual_query_tokens)" in source
