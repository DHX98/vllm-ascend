
def get_global_rank(parallel_config=None):
    try:
        from vllm.distributed.parallel_state import get_tp_group
        return int(get_tp_group().rank)
    except Exception:
        import os
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

# Standard
import functools
import json
import os
import threading
from dataclasses import dataclass
from typing import Any

import regex as re
import torch

# Third Party
from mooncake.store import ReplicateConfig  # type: ignore
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.parallel_state import get_global_rank

DEFAULT_GLOBAL_SEGMENT_SIZE = 1073741824  # 1.0 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB


@functools.lru_cache(maxsize=1)
def _mooncake_setup_supports_ssd_offload() -> bool:
    """True when installed Mooncake exposes SSD kwargs on setup() (v0.3.11+)."""
    from mooncake.store import MooncakeDistributedStore  # type: ignore

    setup = MooncakeDistributedStore.setup
    try:
        import inspect

        sig = inspect.signature(setup)
        return "enable_ssd_offload" in sig.parameters
    except (TypeError, ValueError):
        # pybind11 overloaded bindings often reject inspect.signature
        doc = setup.__doc__ or ""
        return "enable_ssd_offload" in doc



@functools.lru_cache(maxsize=1)
def _mooncake_exist_options_class():
    """Return ExistOptions when the installed Mooncake exposes it."""
    try:
        from mooncake.store import ExistOptions  # type: ignore

        options = ExistOptions()
        if hasattr(options, "prefetch_to_memory"):
            return ExistOptions
    except Exception:
        pass
    return None


@functools.lru_cache(maxsize=1)
def _mooncake_is_exist_supports_prefetch() -> bool:
    """True when installed Mooncake exposes ExistOptions for is_exist / batch_is_exist."""
    return _mooncake_exist_options_class() is not None


def _build_exist_prefetch_options():
    exist_options_cls = _mooncake_exist_options_class()
    if exist_options_cls is None:
        raise RuntimeError("Mooncake ExistOptions is not available")
    options = exist_options_cls()
    options.prefetch_to_memory = True
    return options


def _ssd_setup_kwargs(config: "MooncakeStoreConfig", global_rank: int | None = None) -> dict[str, object]:
    """Keyword args for store.setup(); empty on old Mooncake or when SSD is off."""
    # RANK0_SSD_ONLY: shared TE + identical aclrtMallocHost VA across TP ranks
    # segfaults in FileStorage::RegisterLocalMemory when all ranks enable SSD.
    if not config.enable_ssd_offload:
        return {}
    if global_rank is not None and global_rank != 0:
        logger.warning(
            "RANK0_SSD_ONLY: skip enable_ssd_offload on global_rank=%s to avoid shared-TE register crash",
            global_rank,
        )
        return {}
    if not _mooncake_setup_supports_ssd_offload():
        raise RuntimeError(
            "mooncake.json has enable_ssd_offload=true, but the installed "
            "Mooncake does not support enable_ssd_offload/ssd_offload_path in "
            "MooncakeDistributedStore.setup(). Upgrade Mooncake to v0.3.11 or "
            "later (see Mooncake ssd-offload.md Step 3A), or set "
            "enable_ssd_offload to false."
        )
    kwargs = {
        "enable_ssd_offload": config.enable_ssd_offload,
        "ssd_offload_path": config.ssd_offload_path,
    }
    if _mooncake_is_exist_supports_prefetch():
        if getattr(config, "ssd_prefetch_cooldown_sec", None) is not None:
            kwargs["ssd_prefetch_cooldown_sec"] = config.ssd_prefetch_cooldown_sec
        if getattr(config, "ssd_prefetch_dedup_ttl_sec", None) is not None:
            kwargs["ssd_prefetch_dedup_ttl_sec"] = config.ssd_prefetch_dedup_ttl_sec
    return kwargs


class MooncakeBackend(Backend):
    def __init__(self, parallel_config: ParallelConfig, lazy_init: bool = False, contribute_memory: bool = True):
        self.parallel_config = parallel_config
        self.config = MooncakeStoreConfig.load_from_env()
        if self.config.protocol != "ascend":
            raise NotImplementedError(f"MooncakeBackend does not support protocol {self.config.protocol!r}.")

        self.store: Any | None = None
        self.local_seg: str | None = None
        self._use_fabric_mem = os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") == "1"
        # LAZY_INIT_FOR_SSD: defer MooncakeDistributedStore.setup when SSD is on so
        # Hybrid TransferEngine can finish registerLocalMemory first on rank0.
        self._lazy_init = (lazy_init and self._use_fabric_mem) or bool(self.config.enable_ssd_offload)
        self._contribute_memory = contribute_memory
        self._store_initialized = False
        self._store_init_lock = threading.Lock()
        self._ssd_prefetch_enabled = False
        self._exist_prefetch_options = None

        if not self._lazy_init:
            self.store = self._setup_store()
            self._store_initialized = True

    def ensure_initialized(self):
        if self._store_initialized:
            return

        with self._store_init_lock:
            if self._store_initialized:
                return

            logger.info("Initializing Mooncake store. metadata_server=%s", self.config.metadata_server)
            self.store = self._setup_store()
            self._store_initialized = True

    def _setup_store(self):
        # SKIP_STORE_NONRANK0: only rank0 imports/builds MooncakeDistributedStore
        # when SSD is on. Avoids (1) shared-Hybrid store.setup on other ranks and
        # (2) loading prefetch TE into every TP process.
        _gr = get_global_rank(self.parallel_config)
        # ALL_RANKS_STORE: non-rank0 must also setup so TP shard puts can reach SSD.
        import os as _os
        if self.config.enable_ssd_offload and _gr != 0 and _os.environ.get("MOONCAKE_STORE_RANK0_ONLY") == "1":
            logger.warning(
                "SKIP_STORE_NONRANK0: skip AscendStore store.setup on global_rank=%s",
                _gr,
            )
            self._store_skipped = True
            self.local_seg = get_ip()
            return None
        self._store_skipped = False
        logger.warning("STORE_SETUP_BEGIN global_rank=%s", get_global_rank(self.parallel_config))
        # FORCE_SKIP_FILESTORAGE_TE_REGISTER: ascend host buf RegisterLocalMemory -> bad_alloc
        import os as _os
        _os.environ["MOONCAKE_SKIP_FILESTORAGE_TE_REGISTER"] = "1"
        _os.environ.setdefault("MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES", "16777216")
        _os.environ.setdefault("MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR", "file_per_key_storage_backend")
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        store = MooncakeDistributedStore()
        local_hostname = get_ip()
        ssd_kwargs = _ssd_setup_kwargs(self.config, get_global_rank(self.parallel_config))
        # Each rank that contributes memory to the pool uses its own SSD
        # directory to avoid bucket file collisions. Key by the globally unique
        # rank so that DP/TP/PP/CP replicas never share a directory (dense and
        # MoE alike); only ranks that contribute memory need an offload dir.
        if ssd_kwargs and ssd_kwargs.get("ssd_offload_path") and self._contribute_memory:
            global_rank = get_global_rank(self.parallel_config)
            rank_path = os.path.join(str(ssd_kwargs["ssd_offload_path"]), f"rank_{global_rank}")
            try:
                os.makedirs(rank_path, exist_ok=True)
            except OSError as e:
                raise RuntimeError(f"Failed to create per-rank SSD offload directory: {rank_path!r} ({e})")
            ssd_kwargs["ssd_offload_path"] = rank_path
        # ASCEND_ENABLE_USE_FABRIC_MEM: Enable unified memory address direct transmission scheme
        # and only can be used for 800 I/T A3 series.
        # Required supporting hardware versions are as follows:
        if not self._use_fabric_mem:
            # PRIVATE_TE_FOR_SSD: AscendStore must not share Hybrid TransferEngine
            if ssd_kwargs:
                # SHARE_TE_FOR_SSD: private TE double-installs AscendDirectTransport
                # after Hybrid and aborts with free()/Try-object-empty. Reuse
                # global_te; KEEP FORCE_SKIP to avoid FileStorage RegisterLocalMemory.
                transfer_engine = global_te.get_transfer_engine(local_hostname, device_name=None)
                self.local_seg = local_hostname + ":" + str(transfer_engine.get_rpc_port())
                logger.warning("SHARE_TE_SETUP_FOR_SSD master=%s path=%s seg=%s",
                               self.config.master_server_address,
                               ssd_kwargs.get("ssd_offload_path"),
                               self.local_seg)
                ret = store.setup(
                    local_hostname=self.local_seg,
                    metadata_server=self.config.metadata_server,
                    global_segment_size=0,
                    local_buffer_size=0,
                    protocol=self.config.protocol,
                    rdma_devices=self.config.device_name,
                    master_server_addr=self.config.master_server_address,
                    engine=transfer_engine.get_engine(),
                    enable_ssd_offload=True,
                    ssd_offload_path=str(ssd_kwargs.get("ssd_offload_path", "")),
                )
            else:
                transfer_engine = global_te.get_transfer_engine(local_hostname, device_name=None)
                self.local_seg = local_hostname + ":" + str(transfer_engine.get_rpc_port())
                ret = store.setup(
                    local_hostname=self.local_seg,
                    metadata_server=self.config.metadata_server,
                    global_segment_size=self.config.global_segment_size if self._contribute_memory else 0,
                    local_buffer_size=self.config.local_buffer_size if self._contribute_memory else 0,
                    protocol=self.config.protocol,
                    rdma_devices=self.config.device_name,
                    master_server_addr=self.config.master_server_address,
                    engine=transfer_engine.get_engine(),
                    **ssd_kwargs,
                )
        else:
            self.local_seg = local_hostname
            ret = store.setup(
                local_hostname=self.local_seg,
                metadata_server=self.config.metadata_server,
                global_segment_size=self.config.global_segment_size if self._contribute_memory else 0,
                local_buffer_size=0,
                protocol=self.config.protocol,
                rdma_devices=self.config.device_name,
                master_server_addr=self.config.master_server_address,
                **ssd_kwargs,
            )

        if ret != 0:
            msg = "Initialize mooncake failed."
            logger.error(
                "Initialize mooncake failed. ret=%d, metadata_server=%s. Check mooncake config and network.",
                ret,
                self.config.metadata_server,
            )
            raise RuntimeError(msg)
        if ssd_kwargs:
            logger.info(
                "Mooncake SSD offload enabled (Mode A): path=%s",
                self.config.ssd_offload_path,
            )
        logger.warning("RANK0_STORE_SETUP_OK ret=%s", ret if "ret" in dir() else None)
        return store

    @classmethod
    def create_scheduler_client(cls, parallel_config: ParallelConfig):
        torch.npu.set_device(0)
        return cls(parallel_config, contribute_memory=False)

    def _resolve_ssd_prefetch(self) -> bool:
        """Decide whether to pass ExistOptions(prefetch_to_memory=True) on exist queries."""
        if not getattr(self.config, "enable_ssd_prefetch", False):
            return False
        if not getattr(self.config, "enable_ssd_offload", False):
            logger.warning(
                "enable_ssd_prefetch is true but SSD offload is disabled; "
                "prefetch has no effect without SSD offload."
            )
            return False
        if not _mooncake_is_exist_supports_prefetch():
            logger.warning(
                "enable_ssd_prefetch is true, but the installed Mooncake does "
                "not support ExistOptions on batch_is_exist / is_exist "
                "(RFC #2213). Prefetch will be disabled."
            )
            return False
        logger.info("Mooncake SSD prefetch on exist enabled.")
        return True

    def set_device(self):
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)

    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        if not self._use_fabric_mem:
            local_hostname = get_ip()
            global_te.get_transfer_engine(local_hostname, device_name=None)
            global_te.register_buffer(ptrs, lengths)

    def exists(self, keys: list[str]) -> list[int]:
        if self._lazy_init and not self._store_initialized:
            logger.debug(
                "MooncakeBackend.exists called before store initialization; treating %d keys as missing.",
                len(keys),
            )
            return [0] * len(keys)
        assert self.store is not None
        if self._ssd_prefetch_enabled and self._exist_prefetch_options is not None:
            return self.store.batch_is_exist(keys, self._exist_prefetch_options)
        return self.store.batch_is_exist(keys)

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        self.ensure_initialized()
        assert self.store is not None
        try:
            config = ReplicateConfig()
            if self.config.preferred_segment:
                config.preferred_segment = self.local_seg
            config.prefer_alloc_in_same_node = self.config.prefer_alloc_in_same_node
            res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes, config)
            failed_codes = [int(value) for value in res if value < 0]
            failed_count = len(failed_codes)
            if failed_count:
                error_codes = sorted(set(failed_codes))
                logger.error(
                    "Failed to put %d keys out of %d. error_codes=%s. Check memory and store capacity.",
                    failed_count,
                    len(keys),
                    error_codes,
                )
                logger.debug("Failed to put key details. keys=%s, result=%s", keys, res)
                if self._lazy_init:
                    logger.warning("First DSV4(compress) request failure is expected. This is normal behavior.")
        except Exception as e:
            logger.error(
                "Failed to put %d keys out of %d. type=%s, error=%s. Check store state and memory.",
                len(keys),
                len(keys),
                type(e).__name__,
                e,
            )
            logger.debug("Failed to put key details. keys=%s", keys)
            if self._lazy_init:
                logger.warning("First DSV4(compress) request failure is expected. This is normal behavior.")

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        if self._lazy_init and not self._store_initialized:
            logger.error(
                "Failed to get %d keys out of %d. Store is not initialized; "
                "call put() first to trigger initialization.",
                len(keys),
                len(keys),
            )
            logger.debug("Failed to get key details. keys=%s", keys)
            return
        assert self.store is not None
        logger.debug(
            "MooncakeBackend.get enter keys=%d sample_keys=%s",
            len(keys),
            keys[:3],
        )
        try:
            res = self.store.batch_get_into_multi_buffers(keys, addrs, sizes)
            res_list = list(res)
            failed_codes = [int(value) for value in res_list if value < 0]
            failed_count = len(failed_codes)
            error_codes = sorted(set(failed_codes))
            if failed_count:
                logger.error(
                    "Failed to get %d keys out of %d. error_codes=%s. Check key existence and memory state.",
                    failed_count,
                    len(keys),
                    error_codes,
                )
                logger.debug("Failed to get key details. keys=%s, result=%s", keys, res_list)
            for i, value in enumerate(res_list):
                if value > 0:
                    res_list[i] = 0
            return res_list
        except Exception as e:
            logger.error(
                "Failed to get %d keys out of %d. type=%s, error=%s. Check store state and network.",
                len(keys),
                len(keys),
                type(e).__name__,
                e,
            )
            logger.debug("Failed to get key details. keys=%s", keys)
            return None


@dataclass
class MooncakeStoreConfig:
    metadata_server: str
    global_segment_size: int | str
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    preferred_segment: bool
    prefer_alloc_in_same_node: bool
    enable_ssd_offload: bool = False
    ssd_offload_path: str = ""
    enable_ssd_prefetch: bool = False
    ssd_prefetch_cooldown_sec: int | None = None
    ssd_prefetch_dedup_ttl_sec: int | None = None

    def __post_init__(self) -> None:
        if not self.enable_ssd_offload:
            return
        if not self.ssd_offload_path:
            raise ValueError(
                "enable_ssd_offload is true but ssd_offload_path is empty. Set ssd_offload_path in mooncake.json."
            )
        if not os.path.isabs(self.ssd_offload_path):
            raise ValueError(f"ssd_offload_path must be an absolute path, got: {self.ssd_offload_path!r}")

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        master_server_address = os.getenv("MOONCAKE_MASTER", None)
        global_segment_size_env = os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", None)
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server"),
            global_segment_size=_parse_global_segment_size(
                global_segment_size_env
                if global_segment_size_env is not None
                else config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_global_segment_size(config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)),
            protocol=config.get("protocol", "ascend"),
            device_name=config.get("device_name", ""),
            master_server_address=master_server_address
            if master_server_address is not None
            else config.get("master_server_address"),
            preferred_segment=config.get("preferred_segment", False),
            prefer_alloc_in_same_node=config.get("prefer_alloc_in_same_node", True),
            enable_ssd_offload=bool(config.get("enable_ssd_offload", False)),
            ssd_offload_path=config.get("ssd_offload_path", ""),
            enable_ssd_prefetch=bool(config.get("enable_ssd_prefetch", False)),
            ssd_prefetch_cooldown_sec=(
                int(config["ssd_prefetch_cooldown_sec"])
                if config.get("ssd_prefetch_cooldown_sec") is not None
                else None
            ),
            ssd_prefetch_dedup_ttl_sec=(
                int(config["ssd_prefetch_dedup_ttl_sec"])
                if config.get("ssd_prefetch_dedup_ttl_sec") is not None
                else None
            ),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError("The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_path)


def _parse_global_segment_size(value) -> int:
    """
    Parse storage size strings with support for units: GB, MB, KB, B

    Args:
        value: Input value (int, str, or other convertible types)

    Returns:
        int: Size in bytes

    Raises:
        ValueError: For invalid format, missing number, or negative values
        TypeError: For unsupported input types
    """

    if isinstance(value, int):
        return value
    elif not isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Unsupported type for global_segment_size: {type(value)}") from e

    cleaned_input = value.strip().lower()
    if not cleaned_input:
        raise ValueError("global segment size cannot be empty.")

    UNIT_MULTIPLIERS = {
        "gb": 1024**3,  # 1 GB = 1024^3 bytes
        "mb": 1024**2,  # 1 MB = 1024^2 bytes
        "kb": 1024,  # 1 KB = 1024 bytes
        "b": 1,  # 1 B = 1 byte
    }
    pattern = r"^\s*([\d.]+)\s*(gb|mb|kb|b)?\s*$"
    match = re.match(pattern, cleaned_input)

    if not match:
        raise ValueError(f"Invalid format: '{value}'")

    number_str = match.group(1)
    unit = match.group(2) or "b"

    multiplier = UNIT_MULTIPLIERS[unit]
    return _convert_to_bytes(number_str, multiplier, value)


def _convert_to_bytes(number_str: str, multiplier: int, original_input: str) -> int:
    """
    Convert numeric string to byte count

    Args:
        number_str: Numeric portion of input
        multiplier: Unit conversion factor
        original_input: Original input string (for error messages)

    Returns:
        int: Byte count

    Raises:
        ValueError: For invalid numbers or negative results
    """
    try:
        numeric_value = float(number_str)
    except ValueError:
        raise ValueError(f"Invalid numeric value '{number_str}' in: '{original_input}'")
    # Calculate byte count
    try:
        byte_count = int(numeric_value * multiplier)
    except OverflowError:
        raise ValueError(f"Storage size too large: '{original_input}'")
    return byte_count
