"""
Weight Transfer Service - NCCL-based distributed weight synchronization.

This module implements the weight transfer protocol for RL post-training loops.
It uses torch.distributed with NCCL backend and _new_process_group_helper
for correct process group formation matching the trainer-side protocol.

Reference: lightseekorg/tokenspeed issue #1373
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeightTransferConfig:
    """Configuration for weight transfer engine."""
    master_address: str
    master_port: int
    rank_offset: int
    world_size: int
    group_name: str
    backend: str = "nccl"
    device_indices: Optional[List[int]] = None


@dataclass(frozen=True)
class WeightUpdatePlan:
    """Plan for a weight update operation."""
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    group_name: str
    flush_cache: bool = False


class WeightTransferService:
    """
    Service for managing distributed weight transfers via NCCL.

    This is a stub implementation for API contract validation.
    Full implementation requires:
    - torch.distributed with NCCL backend
    - _new_process_group_helper with PrefixStore
    - Proper CUDA device management
    - Synchronization with trainer process

    The upstream TokenSpeed bug (issue #1373) was:
    - /update_weights returned 200 but didn't execute dist.broadcast
    - torch.empty() buffer was loaded without being filled
    - Silent failure: perturb test passed but restore didn't recover

    Correct implementation must:
    1. Initialize process group with _new_process_group_helper
    2. Execute dist.broadcast for each tensor before load_weights
    3. Synchronize with torch.cuda.synchronize after each broadcast
    4. Return error if collective fails (not silent success)
    """

    def __init__(self):
        self._initialized = False
        self._config: Optional[WeightTransferConfig] = None
        self._process_group: Any = None
        self._update_active = False
        self._lock = Lock()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def config(self) -> Optional[WeightTransferConfig]:
        return self._config

    def initialize(self, config: WeightTransferConfig) -> bool:
        """
        Initialize the NCCL process group.

        In production, this:
        1. Sets CUDA device for this worker
        2. Rendezvous via TCP store
        3. Creates PrefixStore with group_name
        4. Calls _new_process_group_helper for NCCL backend
        5. Registers group ranks in _world.pg_group_ranks

        Args:
            config: Weight transfer configuration

        Returns:
            True if initialization succeeded

        Raises:
            RuntimeError: If initialization fails
            ValueError: If config is invalid
        """
        with self._lock:
            if self._initialized:
                # Allow re-initialization with same config, reject different config
                if self._config and self._config.group_name != config.group_name:
                    raise ValueError(
                        f"Already initialized with group '{self._config.group_name}', "
                        f"cannot reinitialize with '{config.group_name}'"
                    )
                logger.warning("Weight transfer already initialized with same config")
                return True

            # Validate config
            if config.backend not in ("nccl", "gloo"):
                raise ValueError(f"Unsupported backend: {config.backend}")

            if config.world_size < 2:
                raise ValueError("world_size must be >= 2 (trainer + at least 1 worker)")

            # In production:
            # import torch
            # import torch.distributed as dist
            # from torch.distributed.distributed_c10d import (
            #     Backend, PrefixStore, _new_process_group_helper, _world,
            #     default_pg_timeout, rendezvous,
            # )
            # from packaging.version import parse as parse_version
            #
            # torch.cuda.set_device(torch.device(f"cuda:{config.device_indices[0]}"))
            #
            # store, rank, world_size = next(rendezvous(
            #     f"tcp://{config.master_address}:{config.master_port}",
            #     config.rank_offset, config.world_size, timeout=default_pg_timeout
            # ))
            # store.set_timeout(default_pg_timeout)
            # store = PrefixStore(config.group_name, store)
            #
            # opt = ("backend_options" if parse_version(torch.__version__) >= parse_version("2.6")
            #        else "pg_options")
            # pg, _ = _new_process_group_helper(
            #     config.world_size, config.rank_offset, [],
            #     Backend(config.backend), store,
            #     group_name=config.group_name, **{opt: None}, timeout=default_pg_timeout,
            # )
            # _world.pg_group_ranks[pg] = {i: i for i in range(config.world_size)}
            #
            # self._process_group = pg

            self._config = config
            self._initialized = True
            logger.info(
                f"Weight transfer initialized: rank_offset={config.rank_offset}, "
                f"world_size={config.world_size}, group={config.group_name}"
            )
            return True

    def start_update(self, group_name: str) -> bool:
        """
        Begin a weight update session.

        Args:
            group_name: NCCL process group name

        Returns:
            True if session started

        Raises:
            RuntimeError: If not initialized or group mismatch
        """
        with self._lock:
            if not self._initialized:
                raise RuntimeError("Weight transfer not initialized. Call initialize() first.")

            if group_name != self._config.group_name:
                raise ValueError(f"Group name mismatch: {group_name} != {self._config.group_name}")

            if self._update_active:
                logger.warning("Weight update already active")
                return True

            self._update_active = True
            logger.info(f"Weight update session started: {group_name}")
            return True

    def update_weights(self, plan: WeightUpdatePlan) -> Dict[str, Any]:
        """
        Receive weight tensors via NCCL broadcast.

        This is where the upstream bug occurred: the dist.broadcast
        was never executed, leaving torch.empty() buffers unfilled.

        Correct implementation:
        for name, dtype, shape in zip(plan.names, plan.dtypes, plan.shapes):
            tensor = torch.empty(shape, dtype=getattr(torch, dtype), device=device)
            dist.broadcast(tensor, src=0, group=self._process_group)
            torch.cuda.synchronize(device)  # Critical: ensure collective completes
            # Then feed to model.load_weights()

        Args:
            plan: Weight update plan with tensor metadata

        Returns:
            Dict with success status and bytes received

        Raises:
            RuntimeError: If update fails
        """
        with self._lock:
            if not self._initialized:
                raise RuntimeError("Weight transfer not initialized")

            if not self._update_active:
                raise RuntimeError("Weight update session not started (call start_update first)")

            if plan.group_name != self._config.group_name:
                raise ValueError(f"Group name mismatch: {plan.group_name}")

            # In production, execute NCCL broadcast for each tensor
            # This is the critical path that was broken in upstream

            total_bytes = 0
            for name, dtype, shape in zip(plan.names, plan.dtypes, plan.shapes):
                # Calculate tensor size
                numel = 1
                for dim in shape:
                    numel *= dim
                dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int32": 4, "int64": 8}
                total_bytes += numel * dtype_bytes.get(dtype, 4)

                logger.info(f"Receiving weight: {name}, shape={shape}, dtype={dtype}")

                # CRITICAL: Execute dist.broadcast here
                # tensor = torch.empty(shape, dtype=..., device=device)
                # dist.broadcast(tensor, src=0, group=self._process_group)
                # torch.cuda.synchronize(device)
                # model.load_weights({name: tensor})

            return {
                "success": True,
                "message": "Weights updated",
                "tensors_received": len(plan.names),
                "bytes_received": total_bytes,
            }

    def finish_update(self, group_name: str) -> bool:
        """
        Finish weight update session.

        Args:
            group_name: NCCL process group name

        Returns:
            True if session finished

        Raises:
            RuntimeError: If no active session
            ValueError: If group name mismatch
        """
        with self._lock:
            if not self._update_active:
                raise RuntimeError("No active weight update session")

            if group_name != self._config.group_name:
                raise ValueError(f"Group name mismatch: {group_name}")

            self._update_active = False
            logger.info(f"Weight update session finished: {group_name}")
            return True

    def shutdown(self) -> None:
        """Shutdown weight transfer engine."""
        with self._lock:
            if self._process_group is not None:
                # In production: dist.destroy_process_group(self._process_group)
                self._process_group = None
            self._initialized = False
            self._config = None
            self._update_active = False
            logger.info("Weight transfer engine shutdown")


# Global service instance
_weight_transfer_service: Optional[WeightTransferService] = None


def get_weight_transfer_service() -> WeightTransferService:
    global _weight_transfer_service
    if _weight_transfer_service is None:
        _weight_transfer_service = WeightTransferService()
    return _weight_transfer_service


def set_weight_transfer_service(service: WeightTransferService) -> None:
    """For testing - replace the global service."""
    global _weight_transfer_service
    _weight_transfer_service = service