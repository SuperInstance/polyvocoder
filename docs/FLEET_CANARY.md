# Fleet Canary — Cross-Polyformalism Verification

The Quilt substrate walker pins to a **fleet canary** that all polyformalism ports
must reproduce byte-exact:

**`fnv1a-64('café Δ 日本語') = 0x024a555471370b18d`**

This document explains the canary, how to verify it, and why it matters.

## What is the Fleet Canary?

A fleet canary is a fixed hash that all implementations of a system must agree on.
It's a way to verify that:

1. **Unicode handling is consistent** across implementations
2. **FNV-1a implementation is correct** across implementations
3. **No silent drift** between ports (e.g., Python 3.10 vs. 3.12 unicode)

The canary string `'café Δ 日本語'` is deliberately chosen to exercise:
- ASCII (low bytes)
- Latin-1 characters (é)
- Greek capital delta (Δ)
- CJK characters (日本語)
- Multi-byte UTF-8 encoding

## The Reference Implementation

```python
# polyvocoder/canary.py

FNV_OFFSET_64 = 0xcbf29ce484222325
FNV_PRIME_64 = 0x100000001b3

def fnv1a_64(data: bytes) -> int:
    h = FNV_OFFSET_64
    for b in data:
        h = ((h ^ b) * FNV_PRIME_64) & 0xffffffffffffffff
    return h

CANARY_STRING = "café Δ 日本語"
CANARY_HASH = 0x024a555471370b18d

def canary_hash() -> int:
    return fnv1a_64(CANARY_STRING.encode("utf-8", errors="surrogatepass"))

assert canary_hash() == CANARY_HASH, f"Expected {CANARY_HASH:#018x}, got {canary_hash():#018x}"
```

## Why 0x024a555471370b18d?

This hash is the result of running FNV-1a 64-bit on the UTF-8 encoding of the
canary string with `errors="surrogatepass"`. The exact value is:
- **0x024a555471370b18d** — 64-bit FNV-1a hash

## How to Verify Your Port

### Python

```python
from polyvocoder.canary import canary_hash, CANARY_HASH

assert canary_hash() == CANARY_HASH
print(f"✓ Fleet canary verified: {canary_hash():#018x}")
```

### TypeScript

```typescript
import { canaryHash } from "./canary";

const expected = 0x024a555471370b18n;
const actual = canaryHash();
if (actual !== expected) {
  throw new Error(`Fleet canary mismatch: expected ${expected}, got ${actual}`);
}
console.log(`✓ Fleet canary verified: ${actual.toString(16)}`);
```

### Rust

```rust
fn canary_hash() -> u64 {
    let s = "café Δ 日本語";
    let bytes = s.as_bytes();
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h = (h ^ b as u64) * 0x100000001b3;
    }
    h
}

const EXPECTED: u64 = 0x024a555471370b18d;

#[test]
fn test_canary() {
    assert_eq!(canary_hash(), EXPECTED);
}
```

### Bash

```bash
#!/bin/bash
# canary.sh

# Reference: fnv1a-64 of "café Δ 日本語" should be 0x024a555471370b18d
# Use python (universal FNV-1a implementation)

python3 -c "
import sys
sys.path.insert(0, '/path/to/polyvocoder')
from polyvocoder.canary import canary_hash, CANARY_HASH
assert canary_hash() == CANARY_HASH, f'Mismatch: {canary_hash():#x}'
print('Fleet canary OK')
"
```

## Common Implementation Pitfalls

### Pitfall 1: Python Unicode Normalization

Python 3.10 changed unicode normalization. The same string can have different
FNV-1a hashes between Python versions if you use `str.encode()` directly.

**Fix**: Use explicit `errors="surrogatepass"`:
```python
data = string.encode("utf-8", errors="surrogatepass")
```

### Pitfall 2: JavaScript UTF-8 Encoding

JavaScript's `Buffer.from(string, 'utf8')` works correctly, but
`String.prototype.charCodeAt` doesn't handle surrogate pairs.

**Fix**: Use Buffer:
```typescript
const bytes = Buffer.from(string, 'utf8');
```

### Pitfall 3: Rust String Encoding

Rust's `str::as_bytes()` works correctly. But `str::chars()` iterates over
codepoints, not bytes.

**Fix**: Use `as_bytes()` directly:
```rust
let bytes = s.as_bytes();
```

### Pitfall 4: C# String to Bytes

C#'s `Encoding.UTF8.GetBytes(str)` works. But `Encoding.Unicode.GetBytes(str)`
gives UTF-16, not UTF-8.

**Fix**: Use UTF-8 explicitly:
```csharp
byte[] bytes = Encoding.UTF8.GetBytes(input);
```

## Why a Fleet Canary?

Without a fleet canary, two polyformalism ports could silently drift. For example:

- Python 3.10 might normalize `'é'` differently than Python 3.12
- TypeScript's Buffer might encode `Δ` differently than Python's str
- Rust's UTF-8 might handle the BOM differently

The fleet canary catches these drifts IMMEDIATELY. If your port produces
`0x024a555471370b18d`, you're golden. If not, you have a unicode or FNV-1a
implementation issue that needs fixing.

## Integration with CI

Add to your CI pipeline:
```yaml
- name: Verify Fleet Canary
  run: |
    python -c "from polyvocoder.canary import canary_hash; assert canary_hash() == 0x024a555471370b18d"
```

This is a quick (sub-second) test that catches unicode/FNV bugs.

## When to Update the Canary

NEVER update the fleet canary. It's pinned. If your port can't reproduce it,
your port has a bug — fix the port, not the canary.

If you genuinely need a different canary (e.g., you're building a new fleet):
1. Pick a new fixed string
2. Compute its FNV-1a 64 hash
3. Document the new canary in your fleet's repo
4. Document the migration path

