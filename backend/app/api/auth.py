from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, get_db
from app.core.security import (
    PASSWORD_TOO_LONG_MESSAGE,
    PasswordTooLongError,
    create_access_token,
    hash_password,
    verify_password,
)
from app.db.models.user import User
from app.schemas.auth import AuthUserResponse, LoginRequest, RegisterRequest, RegisterResponse, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


def _user_response(user: User) -> AuthUserResponse:
    return AuthUserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=True,
        is_admin=user.role == "admin",
    )


@router.post("/register", response_model=RegisterResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered.")

    try:
        password_hash = hash_password(payload.password)
    except PasswordTooLongError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=PASSWORD_TOO_LONG_MESSAGE)

    user = User(
        email=payload.email,
        full_name=payload.full_name,
        password_hash=password_hash,
        role="user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(str(user.id))
    return RegisterResponse(access_token=token, user=_user_response(user))


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    try:
        password_valid = bool(user and verify_password(payload.password, user.password_hash))
    except PasswordTooLongError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=PASSWORD_TOO_LONG_MESSAGE)

    if not password_valid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(str(user.id))
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
    )
