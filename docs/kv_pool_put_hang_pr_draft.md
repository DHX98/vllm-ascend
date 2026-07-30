# PR Draft — AscendStore Put path hang (`wait_for_save` / `sample_tokens` timeout)

## Title

`[BugFix][KV Pool] Ensure KV transfer threads always task_done on errors (avoid wait_for_save deadlock)`

## Summary

On DeepSeek-V4-Flash (hybrid KV) with `AscendStoreConnector` + Mooncake, exceptions in
`KVCacheTransferThread` workers historically left `request_queue.join()` blocked when
`task_done()` was skipped. That starves the executor shm broadcast ring and surfaces as:

`TimeoutError: RPC call to sample_tokens timed out` / EngineDead

(often near the default `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300`).

Observed on A3 single-node pool deploy (NPU 4–7, dp2 tp4): Put path `IndexError` while
indexing remapped `group_block_hashes` under hybrid `cache_family_ratio`, then hang.

## Community check

| Source | Fixes this hang? | Notes |
|--------|------------------|-------|
| [vllm-ascend#9405](https://github.com/vllm-project/vllm-ascend/issues/9405) | No | Same symptoms; timeout / `max_chunks` advice |
| [vllm-ascend#10828](https://github.com/vllm-project/vllm-ascend/issues/10828) | No | kv_pool timeout; layer_sharding workaround |
| [PR #11349](https://github.com/vllm-project/vllm-ascend/pull/11349) | No (perf) | Per-request save wait — not exception `task_done` |
| [PR #11348](https://github.com/vllm-project/vllm-ascend/pull/11348) | Partial | Lazy grouped hashes / coordinator |

Mainline SendingThread already catches store errors and calls `task_done()` in `finally`,
and put key build already multiplies `cache_family_ratio`. **Remaining gaps this commit
addresses:**

1. Base `KVTransferThread._handle_request_exception` was `pass` — subclasses that do not
   override it still skip `task_done()` when `run()` routes errors there.
2. `run()` error log lacked traceback.
3. `enable_kv_event` path still called `get_block_hashes` with raw `group_block_size`
   (no `cache_family_ratio`), inconsistent with put key remapping.

## Changes

File: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`

- Base `_handle_request_exception`: always `task_done()` (best-effort).
- Base `run()`: log traceback; if `request_data` is unset after error, still try `task_done()`.
- SendingThread kv-event hash list: remap with `infer_cache_family_ratio(cache_family)`.

## Test plan

- [ ] UT: transfer thread whose `_handle_request` raises → `request_queue.join()` returns.
- [ ] UT / manual: `enable_kv_event=True` + hybrid cache family ratio > 1 does not IndexError
      on event hash indexing.
- [ ] Manual (Ascend): DSV4-Flash AscendStore+mooncake, short + long chat HTTP 200; no
      sustained shm_broadcast warnings after Put.

## Author note

Validated operationally on host `90.90.97.27` (pool_dsv4, AscendStore, Mooncake offload)
with local hotfix on an older 0.23.x tree; this branch ports the durable pieces onto
current main.
