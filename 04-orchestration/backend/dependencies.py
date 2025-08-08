from typing import Annotated

from auth import verify_api_key
from engines import get_session
from fastapi import Depends
from sqlmodel import Session

SessionDep = Annotated[Session, Depends(get_session)]
# AuthKeyDep = Annotated[str, Depends(verify_api_key)]