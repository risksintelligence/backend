from fastapi import APIRouter, HTTPException, Depends

from app.services.submissions import add_submission, list_submissions, update_submission_status
from app.core.security import require_reviewer
from app.core.auth import require_judge
from app.services.judging import log_judging_activity, get_judging_log

router = APIRouter(prefix='/api/v1/community', tags=['community'])

@router.get('/submissions')
def get_submissions():
    return {"entries": list_submissions()}

@router.post('/submissions')
def create_submission(payload: dict):
    required = {'title', 'author', 'mission', 'link'}
    if not required.issubset(payload):
        raise HTTPException(status_code=400, detail='Missing fields')
    submission = add_submission(payload)
    return submission

@router.post('/submissions/{submission_id}/status')
def set_status(submission_id: str, payload: dict, _: None = Depends(require_reviewer), judge=Depends(require_judge)):
    status = payload.get('status')
    if status not in {'approved', 'rejected', 'pending'}:
        raise HTTPException(status_code=400, detail='Invalid status')
    try:
        entry = update_submission_status(submission_id, status)
        log_judging_activity(submission_id, judge['sub'], status)
        return entry
    except ValueError:
        raise HTTPException(status_code=404, detail='Submission not found')
