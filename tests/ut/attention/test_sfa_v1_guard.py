from pathlib import Path


def test_row_parallel_linear_registers_default_aclnn_buffers():
    source = Path("vllm_ascend/ops/linear.py").read_text()

    assert "def _ensure_aclnn_input_metadata_buffers" in source
    assert "self.register_buffer(buffer_name, default_value, persistent=False)" in source
    assert "self._ensure_aclnn_input_metadata_buffers()" in source


def test_sfa_o_proj_switch_uses_direct_aclnn_contract():
    source = Path("vllm_ascend/attention/sfa_v1.py").read_text()

    assert "def _get_o_proj_aclnn_params" not in source
    assert "self.o_proj_tp_aclnn_input_scale = self.o_proj.aclnn_input_scale.clone().detach()" in source
