#!/usr/bin/env python3
"""
Encrypt Secrets File (US-027)

Encrypts secrets from .env or JSON file to encrypted format.

Usage:
    python scripts/encrypt_secrets.py --input .env --output config/secrets.enc --key secrets.key
    python scripts/encrypt_secrets.py --input secrets.json --output config/secrets.enc --key secrets.key

Examples:
    # Encrypt .env file
    python scripts/encrypt_secrets.py --input .env --output config/secrets.enc --key secrets.key

    # Encrypt JSON file
    python scripts/encrypt_secrets.py --input config/secrets.json --output config/secrets.enc --key secrets.key

    # Generate new key and encrypt
    python scripts/generate_secrets_key.py > secrets.key
    python scripts/encrypt_secrets.py --input .env --key secrets.key
"""

import argparse
import json
from pathlib import Path

from cryptography.fernet import Fernet
from dotenv import dotenv_values


def load_secrets_from_env(env_file: str) -> dict[str, str]:
    """Load secrets from .env file.

    Args:
        env_file: Path to .env file

    Returns:
        Dictionary of secrets
    """
    secrets = dotenv_values(env_file)
    # Filter out empty values and comments
    return {k: v for k, v in secrets.items() if v and not k.startswith("#")}


def load_secrets_from_json(json_file: str) -> dict[str, str]:
    """Load secrets from JSON file.

    Args:
        json_file: Path to JSON file

    Returns:
        Dictionary of secrets
    """
    with open(json_file) as f:
        return json.load(f)


def encrypt_secrets(
    secrets: dict[str, str],
    output_file: str,
    key: bytes,
) -> None:
    """Encrypt secrets to file.

    Args:
        secrets: Dictionary of secrets
        output_file: Path to output encrypted file
        key: Encryption key
    """
    fernet = Fernet(key)

    # Serialize secrets
    secrets_json = json.dumps(secrets, indent=2).encode()

    # Encrypt
    encrypted_data = fernet.encrypt(secrets_json)

    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encrypted_data)

    print(f"✅ Secrets encrypted to {output_file}")
    print(f"   Total secrets: {len(secrets)}")
    print(f"   Encrypted size: {len(encrypted_data)} bytes")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Encrypt secrets file for production deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encrypt .env file
  python scripts/encrypt_secrets.py --input .env --output config/secrets.enc --key secrets.key

  # Encrypt JSON file
  python scripts/encrypt_secrets.py --input secrets.json --output config/secrets.enc --key secrets.key

  # Generate new key first
  python scripts/generate_secrets_key.py > secrets.key
  python scripts/encrypt_secrets.py --input .env --key secrets.key

Security Notes:
  • Never commit secrets.key or secrets.enc to version control
  • Use different keys for staging and production
  • Store keys in secure location (vault, password manager)
  • Rotate keys periodically
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input file (.env or .json)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="config/secrets.enc",
        help="Output encrypted file (default: config/secrets.enc)",
    )

    parser.add_argument(
        "--key",
        "-k",
        required=True,
        help="Path to encryption key file",
    )

    args = parser.parse_args()

    # Load encryption key
    key_path = Path(args.key)
    if not key_path.exists():
        print(f"❌ Error: Key file not found: {args.key}")
        print()
        print("Generate a key first:")
        print("  python scripts/generate_secrets_key.py > secrets.key")
        exit(1)

    key = key_path.read_bytes()

    # Load secrets
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        exit(1)

    if input_path.suffix == ".json":
        secrets = load_secrets_from_json(args.input)
    else:
        # Assume .env format
        secrets = load_secrets_from_env(args.input)

    if not secrets:
        print("❌ Error: No secrets found in input file")
        exit(1)

    print(f"📁 Loaded {len(secrets)} secrets from {args.input}")

    # Encrypt secrets
    encrypt_secrets(secrets, args.output, key)

    print()
    print("Next steps:")
    print("  1. Store encryption key securely (do not commit)")
    print("  2. Deploy with encrypted secrets:")
    print(f"     SECRETS_MODE=encrypted SECRETS_KEY=$(cat {args.key}) python -m src.app.main")


if __name__ == "__main__":
    main()
