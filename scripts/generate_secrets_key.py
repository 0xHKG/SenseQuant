#!/usr/bin/env python3
"""
Generate Encryption Key for Secrets Management (US-027)

Generates a Fernet encryption key for use with encrypted secrets.

Usage:
    python scripts/generate_secrets_key.py

Output:
    - Prints key to stdout
    - Optionally saves to secrets.key file

Example:
    python scripts/generate_secrets_key.py > secrets.key
    export SECRETS_KEY=$(cat secrets.key)
"""

from cryptography.fernet import Fernet


def main() -> None:
    """Generate and display encryption key."""
    key = Fernet.generate_key()

    print("=" * 70)
    print("Generated Encryption Key for Secrets Management")
    print("=" * 70)
    print()
    print("Encryption Key:")
    print(key.decode())
    print()
    print("⚠️  IMPORTANT: Save this key securely!")
    print()
    print("Security Guidelines:")
    print("  • DO NOT commit this key to version control")
    print("  • Store in secure location (password manager, vault, env var)")
    print("  • Different keys for staging/production")
    print("  • Rotate keys periodically")
    print()
    print("Usage:")
    print("  1. Save key to file:")
    print("     echo '<key>' > secrets.key")
    print()
    print("  2. Encrypt secrets:")
    print("     python scripts/encrypt_secrets.py --key secrets.key")
    print()
    print("  3. Deploy with key:")
    print("     SECRETS_MODE=encrypted SECRETS_KEY=$(cat secrets.key) python -m src.app.main")
    print()


if __name__ == "__main__":
    main()
