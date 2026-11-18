from pydantic import BaseModel, EmailStr, field_validator

from app.core.security import PasswordTooLongError, validate_bcrypt_password_length


class RegisterRequest(BaseModel):
    email: EmailStr
    full_name: str
    password: str

    @field_validator("password")
    @classmethod
    def validate_password_length(cls, value: str) -> str:
        try:
            validate_bcrypt_password_length(value)
        except PasswordTooLongError as exc:
            raise ValueError(str(exc)) from exc
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def validate_password_length(cls, value: str) -> str:
        try:
            validate_bcrypt_password_length(value)
        except PasswordTooLongError as exc:
            raise ValueError(str(exc)) from exc
        return value


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: str


class AuthUserResponse(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    is_active: bool = True
    is_admin: bool = False


class RegisterResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUserResponse
