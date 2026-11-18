from datetime import datetime, timedelta, timezone
from jose import jwt
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

BCRYPT_MAX_PASSWORD_BYTES = 72
PASSWORD_TOO_LONG_MESSAGE = "Password must be at most 72 bytes when encoded as UTF-8."


class PasswordTooLongError(ValueError):
    """Raised when a password exceeds bcrypt's UTF-8 byte limit."""


def validate_bcrypt_password_length(password: str) -> None:
    if len(password.encode("utf-8")) > BCRYPT_MAX_PASSWORD_BYTES:
        raise PasswordTooLongError(PASSWORD_TOO_LONG_MESSAGE)


def hash_password(password: str) -> str:
    validate_bcrypt_password_length(password)
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    validate_bcrypt_password_length(plain_password)
    return pwd_context.verify(plain_password, password_hash)


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)
