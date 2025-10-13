"""Centralized secrets management with encryption support (US-027)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from loguru import logger


class SecretsManager:
    """Centralized secrets management with encryption support.

    Supports two modes:
    - "plain": Load secrets from .env file (development)
    - "encrypted": Load secrets from encrypted file (production)

    Example:
        # Development (plain .env)
        manager = SecretsManager(mode="plain")
        api_key = manager.get_secret("BREEZE_API_KEY")

        # Production (encrypted)
        manager = SecretsManager(mode="encrypted", key_path="secrets.key")
        api_key = manager.get_secret("BREEZE_API_KEY")

        # Encrypt secrets
        manager = SecretsManager(mode="plain")
        key = manager.encrypt_secrets(output_file="config/secrets.enc")
        # Save key securely, do not commit to repo
    """

    def __init__(
        self,
        mode: str = "plain",
        key_path: str | None = None,
        encrypted_file: str = "config/secrets.enc",
    ):
        """Initialize secrets manager.

        Args:
            mode: Secrets mode ("plain" or "encrypted")
            key_path: Path to encryption key file (required for encrypted mode)
            encrypted_file: Path to encrypted secrets file
        """
        self.mode = mode
        self.key_path = key_path
        self.encrypted_file = encrypted_file
        self.secrets: dict[str, str] = {}
        self._load_secrets()

    def _load_secrets(self) -> None:
        """Load secrets from .env or encrypted file."""
        if self.mode == "plain":
            self._load_plain_secrets()
        elif self.mode == "encrypted":
            self._load_encrypted_secrets()
        else:
            raise ValueError(f"Invalid secrets mode: {self.mode}")

    def _load_plain_secrets(self) -> None:
        """Load secrets from .env file."""
        from dotenv import load_dotenv

        load_dotenv()

        # Copy environment variables to secrets dict
        self.secrets = dict(os.environ)

        logger.info("Loaded secrets from .env (plain mode)")

    def _load_encrypted_secrets(self) -> None:
        """Load secrets from encrypted file."""
        secrets_path = Path(self.encrypted_file)

        if not secrets_path.exists():
            raise FileNotFoundError(f"Encrypted secrets file not found: {self.encrypted_file}")

        # Load encryption key
        key = self._load_key()
        fernet = Fernet(key)

        # Decrypt secrets
        try:
            encrypted_data = secrets_path.read_bytes()
            decrypted_data = fernet.decrypt(encrypted_data)
            self.secrets = json.loads(decrypted_data.decode())

            logger.info(
                f"Loaded {len(self.secrets)} secrets from {self.encrypted_file} (encrypted mode)"
            )
        except Exception as e:
            raise ValueError(f"Failed to decrypt secrets: {e}") from e

    def _load_key(self) -> bytes:
        """Load encryption key from file or environment.

        Returns:
            Encryption key bytes

        Raises:
            ValueError: If encryption key not found
        """
        # Try loading from file
        if self.key_path:
            key_file = Path(self.key_path)
            if key_file.exists():
                return key_file.read_bytes()

        # Try loading from environment
        key_env = os.getenv("SECRETS_KEY")
        if key_env:
            return key_env.encode()

        raise ValueError("Encryption key not found (provide key_path or SECRETS_KEY env var)")

    def get_secret(self, key: str, default: str | None = None) -> str | None:
        """Get secret value.

        Args:
            key: Secret key
            default: Default value if key not found

        Returns:
            Secret value or default
        """
        return self.secrets.get(key, default)

    def set_secret(self, key: str, value: str) -> None:
        """Set secret value (in-memory only).

        Args:
            key: Secret key
            value: Secret value
        """
        self.secrets[key] = value
        logger.debug(f"Set secret: {key}")

    def encrypt_secrets(self, output_file: str = "config/secrets.enc") -> bytes:
        """Encrypt current secrets to file and return encryption key.

        Args:
            output_file: Path to output encrypted file

        Returns:
            Encryption key (save securely, do not commit)

        Example:
            manager = SecretsManager(mode="plain")
            key = manager.encrypt_secrets("config/secrets.enc")

            # Save key to secure location
            Path("secrets.key").write_bytes(key)

            # Deploy with key
            # SECRETS_MODE=encrypted SECRETS_KEY=$(cat secrets.key) python -m src.app.main
        """
        # Generate new encryption key
        key = Fernet.generate_key()
        fernet = Fernet(key)

        # Encrypt secrets
        secrets_json = json.dumps(self.secrets).encode()
        encrypted_data = fernet.encrypt(secrets_json)

        # Write to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(encrypted_data)

        logger.info(f"Secrets encrypted to {output_file}")
        logger.warning("Save encryption key securely (do not commit to repo)")

        return key

    def list_keys(self) -> list[str]:
        """List all secret keys (for debugging).

        Returns:
            List of secret keys
        """
        return list(self.secrets.keys())

    def to_dict(self) -> dict[str, Any]:
        """Export secrets as dictionary (for testing).

        Warning: Only use in secure contexts. Do not log or expose.

        Returns:
            Dictionary of secrets
        """
        return dict(self.secrets)
