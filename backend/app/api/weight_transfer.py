"""
Weight Transfer API endpoints for NCCL-based distributed weight synchronization.

These endpoints implement the protocol for RL post-training loops:
1. POST /init_weight_transfer_engine - Initialize NCCL process group
2. POST /start_weight_update - Begin weight update session
3. POST /update_weights - Receive broadcasted weights
4. POST /finish_weight_update - Close weight update session

Reference: lightseekorg/tokenspeed issue #1373
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.models.weight_transfer import (
    WeightTransferInitRequestModel,
    WeightTransferInitResponseModel,
    WeightUpdateStartRequestModel,
    WeightUpdateStartResponseModel,
    WeightUpdateRequestModel,
    WeightUpdateResponseModel,
    WeightUpdateFinishRequestModel,
    WeightUpdateFinishResponseModel,
)
from app.services.weight_transfer import (
    WeightTransferService,
    get_weight_transfer_service,
    WeightTransferConfig,
    WeightUpdatePlan,
)


router = APIRouter(prefix="/v1", tags=["weight-transfer"])


@router.post(
    "/init_weight_transfer_engine",
    response_model=WeightTransferInitResponseModel,
    status_code=status.HTTP_200_OK,
)
async def init_weight_transfer_engine(
    request: WeightTransferInitRequestModel,
    service: WeightTransferService = Depends(get_weight_transfer_service),
) -> WeightTransferInitResponseModel:
    """
    Initialize the NCCL process group for weight updates.

    This must be called before any weight update operations.
    The trainer (rank 0) must be running and listening on master_address:master_port
    before this endpoint is called.

    Rank layout: trainer is rank 0, engine worker i takes rank rank_offset + i.
    World size = trainer (1) + engine workers.
    """
    try:
        config = WeightTransferConfig(
            master_address=request.master_address,
            master_port=request.master_port,
            rank_offset=request.rank_offset,
            world_size=request.world_size,
            group_name=request.group_name,
            backend=request.backend,
            device_indices=request.device_indices,
        )

        success = service.initialize(config)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize weight transfer engine",
            )

        return WeightTransferInitResponseModel(
            success=True,
            message="Weight transfer engine initialized",
            engine_rank=request.rank_offset,
            world_size=request.world_size,
            group_name=request.group_name,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Initialization failed: {str(e)}",
        )


@router.post(
    "/start_weight_update",
    response_model=WeightUpdateStartResponseModel,
    status_code=status.HTTP_200_OK,
)
async def start_weight_update(
    request: WeightUpdateStartRequestModel,
    service: WeightTransferService = Depends(get_weight_transfer_service),
) -> WeightUpdateStartResponseModel:
    """
    Begin a weight update session.

    Must be called after init_weight_transfer_engine and before update_weights.
    """
    try:
        success = service.start_update(request.group_name)

        return WeightUpdateStartResponseModel(
            success=success,
            message="Weight update session started",
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )


@router.post(
    "/update_weights",
    response_model=WeightUpdateResponseModel,
    status_code=status.HTTP_200_OK,
)
async def update_weights(
    request: WeightUpdateRequestModel,
    service: WeightTransferService = Depends(get_weight_transfer_service),
) -> WeightUpdateResponseModel:
    """
    Receive weight tensors via NCCL broadcast.

    The trainer (rank 0) broadcasts each tensor in order.
    This endpoint blocks until all tensors are received.

    CRITICAL: This must execute dist.broadcast for each tensor.
    The upstream bug was silent success without executing the collective.
    """
    try:
        # Validate lengths match
        if not (len(request.names) == len(request.dtypes) == len(request.shapes)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="names, dtypes, and shapes must have equal length",
            )

        plan = WeightUpdatePlan(
            names=request.names,
            dtypes=request.dtypes,
            shapes=request.shapes,
            group_name=request.group_name,
            flush_cache=request.flush_cache,
        )

        result = service.update_weights(plan)

        return WeightUpdateResponseModel(
            success=result["success"],
            message=result["message"],
            tensors_received=result["tensors_received"],
            bytes_received=result["bytes_received"],
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )


@router.post(
    "/finish_weight_update",
    response_model=WeightUpdateFinishResponseModel,
    status_code=status.HTTP_200_OK,
)
async def finish_weight_update(
    request: WeightUpdateFinishRequestModel,
    service: WeightTransferService = Depends(get_weight_transfer_service),
) -> WeightUpdateFinishResponseModel:
    """
    Finish a weight update session.

    Called after all update_weights calls in a session.
    """
    try:
        success = service.finish_update(request.group_name)

        return WeightUpdateFinishResponseModel(
            success=success,
            message="Weight update session finished",
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )