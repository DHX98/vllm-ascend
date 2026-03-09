import sys
from unittest.mock import MagicMock, patch

import torch

from tests.ut.attention.utils import patch_distributed_groups
from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import (AscendLightningIndexerMetadata,
                                         align_topk_indices_to_query_slots)
from vllm.distributed.parallel_state import GroupCoordinator

if 'torch_npu._inductor' not in sys.modules:
    sys.modules['torch_npu._inductor'] = MagicMock()

from vllm_ascend.attention.sfa_v1 import (AscendSFABackend, AscendSFAImpl,
                                          AscendSFAMetadata,
                                          AscendSFAMetadataBuilder,
                                          build_dsa_cp_dual_chunk_swap_segments)
from vllm_ascend.utils import enable_dsa_cp


class TestAscendSFABackend(TestBase):

    def test_get_name(self):
        self.assertEqual(AscendSFABackend.get_name(), "ASCEND_SFA")

    def test_get_builder_cls(self):
        self.assertEqual(AscendSFABackend.get_builder_cls(),
                         AscendSFAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendSFABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        result = AscendSFABackend.get_impl_cls()
        self.assertEqual(result, AscendSFAImpl)


class TestAscendSFAMetadata(TestBase):

    def test_ascend_sfa_metadata_default(self):
        num_actual_tokens = 100
        slot_mapping = torch.randn(100, 4, 1024)
        seq_lens = torch.tensor([30, 50])
        cum_query_lens = torch.tensor([0, 30, 80])
        block_table = torch.randint(0, 100, (100, 4))

        rope_dim = 32
        max_seq_len = int(seq_lens.max().item())
        sin = torch.randn(max_seq_len, rope_dim)
        cos = torch.randn(max_seq_len, rope_dim)

        num_input_tokens = 2
        head_dim = None
        attn_mask = None
        attn_state = AscendAttentionState.ChunkedPrefill

        metadata = AscendSFAMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            cum_query_lens=cum_query_lens,
            block_table=block_table,
            sin=sin,
            cos=cos,
            num_input_tokens=num_input_tokens,
            head_dim=head_dim,
            attn_mask=attn_mask,
            attn_state=attn_state,
        )

        self.assertEqual(metadata.num_actual_tokens, num_actual_tokens)
        self.assertIs(metadata.slot_mapping, slot_mapping)
        self.assertTrue(torch.equal(metadata.seq_lens, seq_lens))
        self.assertTrue(torch.equal(metadata.cum_query_lens, cum_query_lens))
        self.assertIs(metadata.block_table, block_table)
        self.assertIs(metadata.sin, sin)
        self.assertIs(metadata.cos, cos)
        self.assertEqual(metadata.num_input_tokens, num_input_tokens)
        self.assertIs(metadata.head_dim, head_dim)
        self.assertIs(metadata.attn_mask, attn_mask)
        self.assertEqual(metadata.attn_state, attn_state)


class TestAscendSFAMetadataBuilder(TestBase):

    @patch('vllm.distributed.parallel_state._TP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def setUp(self, mock_tp):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()

        self.mock_cfg = MagicMock()

        self.mock_cfg.parallel_config = MagicMock()
        self.mock_cfg.parallel_config.tensor_parallel_size = 1
        self.mock_cfg.parallel_config.prefill_context_parallel_size = 1
        self.mock_cfg.parallel_config.decode_context_parallel_size = 1

        self.mock_cfg.compilation_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config.enable_sp = False

        self.mock_cfg.speculative_config.num_speculative_tokens = 0

        self.patcher = patch("vllm.config.get_current_vllm_config",
                             return_value=self.mock_cfg)
        self.patcher.start()

        # Mock parent class __init__ to avoid complex initialization,
        # but still set the essential attributes that child class needs
        def mock_parent_init(self, kv_cache_spec, layer_names, vllm_config,
                             device, metadata_cls, supports_dcp_with_varlen):
            self.metadata_cls = metadata_cls
            self.kv_cache_spec = kv_cache_spec
            self.model_config = vllm_config.model_config
            self.vllm_config = vllm_config
            self.device = device
            self.chunked_prefill_workspace_size = 128 * 1024
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size,
                 vllm_config.model_config.get_head_size()),
                dtype=vllm_config.model_config.dtype,
                device=device,
            )

        self.parent_init_patcher = patch(
            "vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder.__init__",
            mock_parent_init)
        self.parent_init_patcher.start()

        if hasattr(enable_dsa_cp, "cache_clear"):
            enable_dsa_cp.cache_clear()

    def tearDown(self):
        self.patcher.stop()
        self.parent_init_patcher.stop()

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_default(self):
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        assert builder.device == device
        assert builder.vllm_config == vllm_config

    def test_build_dsa_cp_dual_chunk_swap_segments_prefill(self):
        query_lens = torch.tensor([28], dtype=torch.int32).numpy()
        seq_lens = torch.tensor([28], dtype=torch.int32).numpy()

        seg_q, seg_k, seg_req_indices, padded_lens = build_dsa_cp_dual_chunk_swap_segments(
            query_lens=query_lens,
            seq_lens=seq_lens,
            group_size=8,
            rank=3,
        )

        self.assertTrue((seg_q == torch.tensor([2, 4], dtype=torch.int32).numpy()).all())
        self.assertTrue((seg_k == torch.tensor([8, 26], dtype=torch.int32).numpy()).all())
        self.assertTrue((seg_req_indices == torch.tensor([0, 0], dtype=torch.int32).numpy()).all())
        self.assertTrue((padded_lens == torch.tensor([32], dtype=torch.int32).numpy()).all())

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
    ):
        mock_enable_dsa_cp.return_value = False

        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.num_actual_reqs = 9
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100),
                                                 torch.randn(100))

        metadata = builder.build(
            common_prefix_len=10,
            common_attn_metadata=common_attn_metadata,
        )

        assert isinstance(metadata, AscendSFAMetadata)
        assert metadata.num_actual_tokens == common_attn_metadata.num_actual_tokens
        assert metadata.slot_mapping.shape == (100, 4, 1024)

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_tracks_actual_requests_under_graph_padding(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
    ):
        mock_enable_dsa_cp.return_value = False

        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()
        mock_get_current_vllm_config.return_value = cfg

        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 8
        common_attn_metadata.num_actual_tokens = 1
        common_attn_metadata.num_input_tokens = 8
        common_attn_metadata.query_start_loc = torch.tensor(
            [0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32)
        common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32)
        common_attn_metadata.num_actual_reqs = 1
        common_attn_metadata.slot_mapping = torch.arange(8, dtype=torch.int32)
        common_attn_metadata.seq_lens = torch.tensor([1] * 8, dtype=torch.int32)
        common_attn_metadata.seq_lens_cpu = torch.tensor([1] * 8, dtype=torch.int32)
        common_attn_metadata.positions = torch.arange(8, dtype=torch.int64)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.DecodeOnly
        common_attn_metadata.block_table_tensor = torch.zeros((8, 1), dtype=torch.int32)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.lightning_indexer_metadata = None

        mock_get_cos_and_sin_mla.return_value = (
            torch.randn(8, 1, dtype=torch.float16),
            torch.randn(8, 1, dtype=torch.float16),
        )

        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

        assert metadata.num_actual_seqs == 1
        assert metadata.num_local_query_tokens == 1
        assert torch.equal(metadata.cum_query_lens, common_attn_metadata.query_start_loc[1:9])

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_uses_unpadded_block_table_for_skip_metadata(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
    ):
        mock_enable_dsa_cp.return_value = False

        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()
        mock_get_current_vllm_config.return_value = cfg

        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)
        builder.enable_lightning_indexer_skip = True

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 8
        common_attn_metadata.num_actual_tokens = 1
        common_attn_metadata.num_input_tokens = 8
        common_attn_metadata.query_start_loc = torch.tensor(
            [0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32)
        common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32)
        common_attn_metadata.num_actual_reqs = 1
        common_attn_metadata.slot_mapping = torch.arange(8, dtype=torch.int32)
        common_attn_metadata.seq_lens = torch.tensor([1] * 8, dtype=torch.int32)
        common_attn_metadata.seq_lens_cpu = torch.tensor([1] * 8, dtype=torch.int32)
        common_attn_metadata.positions = torch.arange(8, dtype=torch.int64)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.DecodeOnly
        common_attn_metadata.block_table_tensor = torch.arange(16, dtype=torch.int32).view(8, 2)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.lightning_indexer_metadata = AscendLightningIndexerMetadata(
            li_reorder_indices=torch.tensor([0], dtype=torch.int32),
            li_cum_query_lens=torch.tensor([0, 1], dtype=torch.int32),
            li_cum_query_lens_cpu=torch.tensor([0, 1], dtype=torch.int32),
            li_seq_lens=torch.tensor([1, 1], dtype=torch.int32),
            li_seq_lens_cpu=torch.tensor([1, 1], dtype=torch.int32),
            li_skip_request_mask=torch.tensor([True], dtype=torch.bool),
            top_k_indices_of_skipped_queries=torch.full((1, 1, 2048), -1, dtype=torch.int32),
            num_actual_reqs=1,
        )

        mock_get_cos_and_sin_mla.return_value = (
            torch.randn(8, 1, dtype=torch.float16),
            torch.randn(8, 1, dtype=torch.float16),
        )

        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

        assert metadata.num_actual_seqs == 1
        assert metadata.num_local_indexer_tokens == 0
        assert metadata.num_local_query_tokens == 1
        assert metadata.block_table.shape == (2, 2)
        assert torch.equal(metadata.cum_query_lens, torch.tensor([0, 1], dtype=torch.int32))
        assert torch.equal(metadata.seq_lens, torch.tensor([1, 1], dtype=torch.int32))
        assert torch.equal(metadata.block_table[0], common_attn_metadata.block_table_tensor[0])
        assert torch.equal(metadata.block_table[1], common_attn_metadata.block_table_tensor[0])

    @patch("vllm_ascend.attention.sfa_v1.get_tp_group")
    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    def test_ascend_sfa_metadata_builder_tracks_local_query_slots_for_dsa_cp_graph_padding(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
        mock_get_tp_group,
    ):
        mock_enable_dsa_cp.return_value = True

        tp_group = MagicMock()
        tp_group.world_size = 2
        tp_group.rank_in_group = 1
        mock_get_tp_group.return_value = tp_group

        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()
        mock_get_current_vllm_config.return_value = cfg

        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)
        builder.enable_lightning_indexer_skip = True

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 1
        common_attn_metadata.num_actual_tokens = 1
        common_attn_metadata.num_input_tokens = 4
        common_attn_metadata.query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
        common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 1], dtype=torch.int32)
        common_attn_metadata.num_actual_reqs = 1
        common_attn_metadata.slot_mapping = torch.arange(4, dtype=torch.int32)
        common_attn_metadata.seq_lens = torch.tensor([1], dtype=torch.int32)
        common_attn_metadata.seq_lens_cpu = torch.tensor([1], dtype=torch.int32)
        common_attn_metadata.positions = torch.arange(4, dtype=torch.int64)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.DecodeOnly
        common_attn_metadata.block_table_tensor = torch.zeros((1, 1), dtype=torch.int32)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.lightning_indexer_metadata = AscendLightningIndexerMetadata(
            li_reorder_indices=torch.tensor([0], dtype=torch.int32),
            li_cum_query_lens=torch.tensor([0, 1], dtype=torch.int32),
            li_cum_query_lens_cpu=torch.tensor([0, 1], dtype=torch.int32),
            li_seq_lens=torch.tensor([1, 1], dtype=torch.int32),
            li_seq_lens_cpu=torch.tensor([1, 1], dtype=torch.int32),
            li_skip_request_mask=torch.tensor([True], dtype=torch.bool),
            top_k_indices_of_skipped_queries=torch.full((1, 1, 2048), -1, dtype=torch.int32),
            num_actual_reqs=1,
        )

        mock_get_cos_and_sin_mla.return_value = (
            torch.randn(4, 1, dtype=torch.float16),
            torch.randn(4, 1, dtype=torch.float16),
        )

        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

        assert metadata.num_local_query_tokens == 0
        assert metadata.num_local_query_slots == 2
        assert metadata.cos.shape[0] == 2
        assert metadata.slot_mapping.shape[0] == 4

    def test_align_topk_indices_to_query_slots_pads_dummy_local_slots(self):
        topk_indices = torch.arange(2048, dtype=torch.int32).view(1, 1, 2048)

        aligned = align_topk_indices_to_query_slots(topk_indices, 2)

        assert aligned.shape == (2, 1, 2048)
        assert torch.equal(aligned[0], topk_indices[0])
        assert torch.all(aligned[1] == -1)

    @patch("vllm_ascend.attention.sfa_v1.get_tp_group")
    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    def test_ascend_sfa_metadata_builder_regenerates_skip_topk_for_local_dsa_cp_segments(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
        mock_get_tp_group,
    ):
        mock_enable_dsa_cp.return_value = True

        tp_group = MagicMock()
        tp_group.world_size = 2
        tp_group.rank_in_group = 1
        mock_get_tp_group.return_value = tp_group

        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()
        mock_get_current_vllm_config.return_value = cfg

        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)
        builder.enable_lightning_indexer_skip = True

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 1
        common_attn_metadata.num_actual_tokens = 4
        common_attn_metadata.num_input_tokens = 4
        common_attn_metadata.query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
        common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 4], dtype=torch.int32)
        common_attn_metadata.num_actual_reqs = 1
        common_attn_metadata.slot_mapping = torch.arange(4, dtype=torch.int32)
        common_attn_metadata.seq_lens = torch.tensor([4], dtype=torch.int32)
        common_attn_metadata.seq_lens_cpu = torch.tensor([4], dtype=torch.int32)
        common_attn_metadata.positions = torch.arange(4, dtype=torch.int64)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.DecodeOnly
        common_attn_metadata.block_table_tensor = torch.zeros((1, 1), dtype=torch.int32)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.lightning_indexer_metadata = AscendLightningIndexerMetadata(
            li_reorder_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            li_cum_query_lens=torch.tensor([0, 4], dtype=torch.int32),
            li_cum_query_lens_cpu=torch.tensor([0, 4], dtype=torch.int32),
            li_seq_lens=torch.tensor([4, 4], dtype=torch.int32),
            li_seq_lens_cpu=torch.tensor([4, 4], dtype=torch.int32),
            li_skip_request_mask=torch.tensor([True], dtype=torch.bool),
            top_k_indices_of_skipped_queries=torch.tensor(
                [[[0, -1, -1, -1], [0, 1, -1, -1], [0, 1, 2, -1], [0, 1, 2, 3]]],
                dtype=torch.int32,
            ).view(4, 1, 4),
            num_actual_reqs=1,
        )

        mock_get_cos_and_sin_mla.return_value = (
            torch.randn(4, 1, dtype=torch.float16),
            torch.randn(4, 1, dtype=torch.float16),
        )

        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

        expected_local_topk = torch.full((2, 1, 2048), -1, dtype=torch.int32)
        expected_local_topk[0, 0, :3] = torch.tensor([0, 1, 2], dtype=torch.int32)
        expected_local_topk[1, 0, :4] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

        assert metadata.num_local_indexer_tokens == 0
        assert metadata.num_local_query_tokens == 2
        assert metadata.num_local_query_slots == 2
        assert metadata.top_k_indices_skip_li_query.shape == (2, 1, 2048)
        assert torch.equal(metadata.top_k_indices_skip_li_query, expected_local_topk)

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build_for_graph_capture(
            self, mock_get_cos_and_sin_mla, mock_get_current_vllm_config):
        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg

        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.num_actual_reqs = 9
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100),
                                                 torch.randn(100))

        attn_metadata = builder.build_for_graph_capture(
            common_attn_metadata=common_attn_metadata,
            attn_state=AscendAttentionState.DecodeOnly,
        )

        assert isinstance(attn_metadata, AscendSFAMetadata)
        assert attn_metadata.attn_state == AscendAttentionState.DecodeOnly


def test_init_o_proj_tp_full_params_without_aclnn_params():
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.tp_size = 2
    impl.o_proj = MagicMock()
    impl.o_proj.weight = torch.randn(4, 8)

    origin_pool = AscendSFAImpl.o_proj_full_pool
    AscendSFAImpl.o_proj_full_pool = None
    try:
        impl._init_o_proj_tp_full_params()
    finally:
        AscendSFAImpl.o_proj_full_pool = origin_pool

    assert impl.o_proj_has_aclnn_params is False
    assert hasattr(impl, "o_proj_tp_weight")
