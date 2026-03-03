# DSA CP 演进方案分析（方案2：保留 FlashComm1）

## 1. 背景与目标

面向 DSA 稀疏注意力场景（DeepSeek V3.2 / GLM5），本方案选择“保留 FlashComm1，并额外实现 Dual Chunk Swap 负载均衡”，目标是同时满足以下诉求：

1. Attention 阶段维持 `TP=1` 语义（逻辑上按 token 切分）并最大化 TTFT 收益。
1. 支持 KV Cache 去冗余以承载长序列，同时不引入明显 TTFT 劣化。
1. 降低 Lightning Indexer 在长序列中的占比，拿到预期约 10% TTFT 收益。
1. 混部场景保留 CP 的 TTFT 优势，且 TPOT 相比 PD 分离不显著退化。

## 2. 本地代码现状（与方案2直接相关）

### 2.1 DSA-CP 在 SFA 中仍是“连续切片”

当前 `AscendSFAMetadataBuilder.build` 在 DSA-CP 分支里，对 token 做的是按 TP rank 的连续区间切分：

- 入口：`vllm_ascend/attention/sfa_v1.py`
- 关键逻辑：`local_start/local_end_with_pad` + `slot_mapping_cp = slot_mapping[local_start:local_end_with_pad]`
- 对应 `actual_seq_lengths_query/actual_seq_lengths_key` 也是基于“连续区间覆盖”构造

这与 Dual Chunk Swap 的“首尾双段分配”不同，无法对齐 Lightning Indexer 负载均衡目标。

### 2.2 PCP 已有 Dual Chunk Swap 能力

`PCPManager.update_tokens_for_pcp` 已实现 DualChunkSwap（按 `2 * pcp_world_size` 切分 + 首尾配对）：

- 文件：`vllm_ascend/worker/pcp_utils.py`
- 已有能力：
    - `pcp_tokens/positions` 计算
    - `pcp_allgather_restore_idx` 恢复索引
    - `num_pcp_pads_cpu` 与 `pcp_unpad_mask` 管理

这部分可以抽取为“通用 token 重排 planner”，复用于 DSA-CP（TP 域）而不是只服务 PCP（PCP 域）。

### 2.3 KV 去冗余已有可复用实现形态

SFA 的 CP 实现里已经有跨 CP 聚合 KV 的模式：

- 文件：`vllm_ascend/attention/context_parallel/sfa_cp.py`
- 关键函数：`gather_kv_cross_cp`
    - 对 `kv_cache` 先 `index_select(valid_block_ids)` 再 `all_gather(dcp)`、`all_gather(pcp)`
- `block_table_cp` 扩展方式也已存在（可复用到 DSA-CP + DCP）。

### 2.4 model_runner 已具备“prepare_inputs 后再重排”的接入点

`NPUModelRunner` 的输入构建和 metadata 构建路径稳定，可用于接入“DSA-CP token 重排 + 恢复”：

- 文件：`vllm_ascend/worker/model_runner_v1.py`
- 关键位置：
    - `prepare_inputs` 中 `num_scheduled_tokens/positions/slot_mapping` 生成
    - `_build_attention_metadata` 中注入 `AscendCommonAttentionMetadata`
    - forward 后已有 PCP `get_restore_hidden_states` 流程，可并行扩展 DSA 恢复流程

## 3. PR #6667 的可复用思路

参考 PR：<https://github.com/vllm-project/vllm-ascend/pull/6667>

从 PR #6667 可提炼出 4 个可直接复用的方向（不要求 1:1 搬运实现）：

1. 在 `attention/utils.py` 增加针对 Lightning Indexer 的元数据封装（重排索引、跳过信息）。
1. 在 `sfa_v1.py` metadata builder 中做“pad + reorder + num_actual_reqs 截断”的统一构建。
1. 在 `sfa_v1.py` 前向中加入“indexer 前重排 / skip 分支拼接 topk”的组合流程。
1. 通过 `additional_config.lightning_indexer_skip` 开关控制，避免新增环境变量。

结论：PR #6667 已验证“重排 + indexer skip”的工程骨架可行，方案2可在此骨架上叠加 Dual Chunk Swap + DCP KV 去冗余。

## 3.1 GLM5 可用性约束（必须满足）

为确保方案对 GLM5（`model_type=glm_moe_dsa`）可用，落地时需要显式保留以下行为：

1. 保留 GLM 分支的 indexer 与 RoPE 路径：
   - `is_rope_neox_style=False`
   - `use_torch_npu_lightning_indexer=True`
   - 即 GLM5 继续走 `torch_npu.npu_lightning_indexer` 分支，不强制切到 `torch.ops._C_ascend.npu_lightning_indexer`。
1. Dual Chunk Swap 的重排逻辑只处理 token 顺序，不改变模型分支判定条件；GLM5 与 DeepSeek 共用 planner，但各自保留算子调用分支。
1. `enable_dsa_cp()` 判定不能写成 DeepSeek 专属条件；应继续依赖“DSA 能力 + SP 开关”而非写死模型名。
1. `skip_threshold` 需可配置（建议 `additional_config.lightning_indexer_skip_threshold`），避免 GLM5 长上下文下固定 2048 阈值不最优。

## 4. 方案2总体设计

### 4.1 设计原则

1. 不删除 FlashComm1，保留 TP16/现网并行形态与上线路径。
1. 将“Dual Chunk Swap 负载均衡”和“KV 去冗余（DCP）”解耦，按开关独立启用。
1. 只在 Attention 相关链路改动，尽量不侵入 Embedding/MLP/LMHead 并行策略。
1. 先做可验证的功能闭环，再逐步做多流掩盖优化。

### 4.2 目标执行流（Prefill）

1. `prepare_inputs` 生成原始 `input_ids/positions/query_lens/slot_mapping`。
1. DSA-CP planner 在 TP 域做 Dual Chunk Swap，输出：
   - `reorder_idx`
   - `restore_idx`
   - `per_req_split_lens`
   - `actual_seq_lengths_{query,key}`（适配算子）
1. 对输入做“重排 + 必要 pad”，形成“Indexer 前缀 + skip 后缀”布局：
   - 前缀：`[req_i(>skip_threshold) 的 head/tail 双段]`
   - 后缀：`[req_i(<=skip_threshold) 或每请求短段]`
1. Lightning Indexer 仅计算前缀部分；后缀直接由规则生成 topk（skip 分支）。
1. 合并 topk 后执行 Sparse Flash Attention。
1. 若开启 DCP 去冗余，则在 Indexer/SFA 前对 `kv_cache[0..2]` 按 DCP 域 all-gather，并扩展 `block_table`。

## 5. 详细改造点（按文件）

## 5.1 `vllm_ascend/attention/utils.py`

新增或扩展 Lightning Indexer 元数据（命名可沿用 PR #6667）：

- `reorder_indices`
- `restore_indices`
- `num_no_skip_reqs`
- `num_actual_reqs`
- `skip_threshold`（默认 2048）
- `top_k_indices_skip_li_query`（可缓存）

新增工具函数：

1. `get_index_of_skipped_queries_numpy(...)`
1. `get_sfa_skip_indices(...)`
1. `maybe_pad_and_reorder_inputs(...)`
1. `hidden_states_reorder(...)`

说明：优先走 `additional_config` 开关，不新增 `envs.py` 变量，规避环境变量评审成本。

## 5.2 `vllm_ascend/worker/pcp_utils.py`

把 DualChunkSwap 的纯算法部分抽成可复用 helper（TP/PCP 均可用）：

- 输入：`query_lens`、`group_size`、`rank`、`decode_threshold`
- 输出：`local_positions`、`restore_idx`、`pads`、`unpad_mask`

保留现有 PCP 路径不变，DSA-CP 调用同一套 planner，避免两套算法漂移。

## 5.3 `vllm_ascend/worker/model_runner_v1.py`

在 `prepare_inputs` 与 `_build_attention_metadata` 注入 DSA-CP planner 结果：

1. `prepare_inputs`：
   - 在 token 选取完成后，按 planner 对 `input_ids/positions` 做重排（仅 DSA-CP + prefill 生效）。
1. `_build_attention_metadata`：
   - 把 planner 输出挂到 `AscendCommonAttentionMetadata`（如 `lightning_indexer_metadata`）。
1. 前向输出后：
   - 对需要恢复顺序的张量执行 `hidden_states_reorder`，保证采样和下游逻辑不变。

## 5.4 `vllm_ascend/attention/sfa_v1.py`

### Metadata Builder

在 `AscendSFAMetadataBuilder.build` 中：

1. 读取并应用 `lightning_indexer_metadata`。
1. 生成 Dual Chunk Swap 后的 `slot_mapping_cp` 与 `actual_seq_lengths_*`。
1. 用 `num_actual_reqs` 截断 `actual_seq_lengths_*`，避免内外 shape 不一致。
1. 持有 skip 分支所需 `top_k_indices_skip_li_query`。

### Forward / Indexer

在 `AscendSFAImpl.forward` 和 `indexer_select_post_process` 中：

1. Indexer 前重排 `hidden_states/q/qr`，保证 query 与 metadata 对齐。
1. 前缀调用 `npu_lightning_indexer`，后缀走 skip topk 生成。
1. 拼接前缀/后缀 topk，必要时 pad 回图模式 shape。
1. 若启用 DCP 去冗余：
   - 对 `kv_cache[2]` 做 all-gather 后再喂 Indexer（注意使用 gather 后的 key，而非本地 key）。
   - 对 `kv_cache[0..1]` 做 all-gather 后喂 SFA。

补充（GLM5）：

1. `indexer_select_post_process` 里的 skip 分支合并要同时兼容两种返回签名：
   - `torch_npu.npu_lightning_indexer -> (topk_indices, ...)`
   - `torch.ops._C_ascend.npu_lightning_indexer -> topk_indices`
1. 重排前后都不能改写 `is_rope_neox_style` 语义，GLM5 仍按非 neox RoPE 运行。

## 5.5 `vllm_ascend/attention/context_parallel/sfa_cp.py`（复用）

复用已有 `gather_kv_cross_cp` 与 `block_table_cp` 扩展逻辑，落到 DSA-CP 路径时建议抽公共函数：

- `expand_block_table_for_cp(block_table, valid_block_ids, cp_size)`
- `gather_kv_by_valid_blocks(kv_cache, valid_block_ids, dcp_group)`

## 6. KV Cache 去冗余（方案2视角）

## 6.1 block_table 扩展规则

若 DCP size = `C`，单卡有效块数为 `B`，则每个 block id 扩展为：

`[id + 0*B, id + 1*B, ..., id + (C-1)*B]`

再按请求内原顺序拼接，作为算子 `block_table` 入参。

## 6.2 通信顺序

1. Indexer 前：all-gather `kv_cache[2]`。
1. SFA 前：all-gather `kv_cache[0]`、`kv_cache[1]`。
1. 本地 cache 更新与全局临时 cache 更新分离，优先保障主流可尽早消费。

## 7. 多流并行优化设计

建议引入专用 `cp_comm_stream`（可复用现有 `global_stream/cp_chunkedprefill_comm_stream` 风格）：

1. 主流：Q/weights/indexer/SFA 主计算。
1. 从流：KV all-gather + 必要向量操作。
1. 同步点：仅在算子真正消费 gathered KV 前等待 event。

目标：减少主流等待，改善 TTFT，避免 decode 阶段通信放大 TPOT。

## 8. 分阶段实施计划

## Phase 1：DualChunkSwap + Indexer Skip（无 DCP）

1. 落地重排元数据与 skip 分支。
1. 验证功能正确性和 TTFT 收益。
1. 保持 decode 路径最小改动，先不引入 KV 去冗余通信。

## Phase 2：对接 DCP KV 去冗余

1. 接入 `kv_cache[0..2]` gather + `block_table` 扩展。
1. 打通长序列和 PD 分离场景。
1. 引入开关，混部默认可关闭 decode 侧去冗余通信。

## Phase 3：多流掩盖与图模式稳态

1. 通信流与计算流重叠优化。
1. 优化图捕获 shape 稳定性与 pad 策略。
1. 完整性能回归（TTFT/TPOT/吞吐/显存）。

## 9. 测试与验收

## 9.1 单测（UT）

新增或扩展：

1. `tests/ut/worker/test_pcp_manager.py`
   - DualChunkSwap planner 在 TP 域的输出正确性（positions/restore_idx/unpad_mask）。
1. `tests/ut/attention/test_sfa_v1.py`
   - DSA-CP metadata 构建后的 `actual_seq_lengths_*`、`num_actual_reqs` 截断正确性。
   - Indexer skip 前缀/后缀拼接 topk 正确性。
1. `tests/ut/worker/test_block_table.py`
   - DCP 去冗余下 block_table 扩展与 slot_mapping 一致性。

## 9.2 集成测试（E2E）

建议覆盖：

1. 单机 DeepSeek V3.2（`TP=16`）长短混合请求，验证 TTFT。
1. PD 分离：P 节点开启 DCP 去冗余，D 节点不开启，验证 TPOT 不退化。
1. 混部：CP 开启/关闭 AB 对比，验证吞吐与尾延迟。
1. 单机 GLM5（A2: `TP=8`，A3: `TP=16`）开启 FlashComm1 + DSA-CP，验证功能正确性与 TTFT。
1. GLM5 长序列（32k/64k+）下 sweep `lightning_indexer_skip_threshold`，选取 TTFT 最优点并回归精度。

## 9.3 关键验收指标

1. TTFT：相对现网基线提升（目标与业务基线对齐，优先验证 10% 级别 indexer 负载收益）。
1. TPOT：混部场景相对 PD 分离不可出现显著退化（建议门限 <5%）。
1. 显存：开启 DCP 去冗余后，长序列可承载能力提升。
1. 正确性：与未开启方案的输出一致性通过阈值校验。

## 10. 风险与规避

1. 风险：重排后 query/key 对不齐，导致精度回退。  
   规避：统一使用 `reorder_idx/restore_idx`，并在 UT 校验 token 级映射。

1. 风险：图模式下动态 shape 抖动导致重捕获。  
   规避：固定 pad 策略，`num_actual_reqs` 与算子输入分离。

1. 风险：DCP gather 增加 decode 通信开销。  
   规避：去冗余开关分阶段启用，混部默认 decode 侧保守策略。

1. 风险：实现分叉（PCP 与 DSA-CP 两套 DualChunkSwap）。  
   规避：把算法层抽成共用 helper，减少后续维护成本。
1. 风险：GLM5 分支误用 DeepSeek 的 indexer/rope 逻辑导致精度劣化。  
   规避：在 `model_type=glm_moe_dsa` 上保留现有分支，并增加 GLM5 专项 UT/E2E。

## 11. 结论

方案2在当前代码基线上可落地，且与 PR #6667 的“重排 + indexer skip”框架天然兼容。推荐采用“先负载均衡、后去冗余、再多流优化”的三阶段推进路径，以最小风险拿到 TTFT 收益并控制混部 TPOT。
