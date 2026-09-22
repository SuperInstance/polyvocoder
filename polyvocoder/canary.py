"""Fleet canary verification.

The polyvocoder pins to the Quilt substrate walker fleet canary:
    fnv1a-64('café Δ 日本語') = 0x024a555471370b18d

This must be byte-exact across all polyformalism ports.
"""

FNV_OFFSET_64 = 0xcbf29ce484222325
FNV_PRIME_64 = 0x100000001b3

CANARY_STRING = "café Δ 日本語"
CANARY_HASH = 0x024a555471370b18d


def fnv1a_64(data: bytes) -> int:
    """FNV-1a 64-bit hash."""
    h = FNV_OFFSET_64
    for b in data:
        h = ((h ^ b) * FNV_PRIME_64) & 0xffffffffffffffff
    return h


def canary_hash() -> int:
    """Compute the fleet canary hash."""
    return fnv1a_64(CANARY_STRING.encode("utf-8", errors="surrogatepass"))


def verify_canary() -> bool:
    """Returns True if the canary matches expected."""
    return canary_hash() == CANARY_HASH


if __name__ == "__main__":
    h = canary_hash()
    if verify_canary():
        print(f"✓ Fleet canary verified: 0x{h:016x}")
    else:
        print(f"✗ Fleet canary MISMATCH: 0x{h:016x} (expected 0x{CANARY_HASH:016x})")
        raise SystemExit(1)
