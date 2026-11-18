from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.security import (
    PasswordTooLongError,
    hash_password,
    validate_bcrypt_password_length,
    verify_password,
)


def test_validate_short_password_passes():
    validate_bcrypt_password_length("shortpass123")


def test_hash_and_verify_short_password_succeeds():
    password_hash = hash_password("shortpass123")

    assert verify_password("shortpass123", password_hash) is True


def test_validate_exactly_72_ascii_bytes_passes():
    validate_bcrypt_password_length("a" * 72)


def test_validate_73_ascii_bytes_fails():
    with pytest.raises(PasswordTooLongError):
        validate_bcrypt_password_length("a" * 73)


def test_validate_multibyte_password_at_72_bytes_passes():
    password = "\u00e9" * 36

    assert len(password.encode("utf-8")) == 72
    validate_bcrypt_password_length(password)


def test_validate_multibyte_password_over_72_bytes_fails():
    password = "\u00e9" * 37

    assert len(password.encode("utf-8")) == 74
    with pytest.raises(PasswordTooLongError):
        validate_bcrypt_password_length(password)
