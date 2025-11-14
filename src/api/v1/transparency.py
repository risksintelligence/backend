from fastapi import APIRouter, Depends

from backend.src.services.transparency_service import TransparencyService, get_transparency_service

router = APIRouter(prefix="/api/v1/transparency", tags=["transparency"])


@router.get("")
async def get_transparency(service: TransparencyService = Depends(get_transparency_service)):
    return service.build_payload()
