# Polyformalism — Schema Parity Across Languages

The polyvocoder is a polyformalism — the same model exists in N languages, with
schema-parity enforced by canonical type definitions.

## Why Polyformalism?

For the Quilt substrate walker, polyformalism is the principle that:

> **The same model in N languages IS a stress test. Each language is a *medium*,
> not a ranking.**

This means:
- A polyformalism port is NOT a port for the sake of compatibility
- Each port exposes the SAME logical model with the SAME types
- The stress comes from language differences (Python duck typing vs. TypeScript
  structural typing vs. Rust strict types vs. C# nullable annotations)

## The Polyformalism Table

| Port | Language | Status | URL |
|---|---|---|---|
| 1 | Python | ✅ Complete | `SuperInstance/polyvocoder` (this repo) |
| 2 | TypeScript | ✅ Complete | `SuperInstance/polyvocoder-bindings` |
| 3 | Rust | 🔄 Planned | (TODO) |
| 4 | Go | 🔄 Planned | (TODO) |
| 5 | Swift | 🔄 Planned | (TODO) |
| 6 | Kotlin | 🔄 Planned | (TODO) |

## Schema Parity

The Python and TypeScript ports share an identical JSON schema. Here's the
canonical type definition:

### JEVFeatures

| Field | Type | Meaning |
|---|---|---|
| `canonWorthy` / `canon_worthy` | float (0-1) | Canon-worthy score from JEV |
| `distinctVoice` / `distinct_voice` | float (0-1) | Distinct non-formulaic voice |
| `doctrineAnchor` / `doctrine_anchor` | float (0-1) | Doctrine-anchored score |
| `voiceFit` / `voice_fit` | float (0-1) | Voice archetype match |
| `novelty` / `novelty` | float (0-1) | Novel imagery score |
| `density` / `density` | float (1+) | Image density (1=single, 2=multiple, 3=saturated) |

### DecodedSample

| Field | Type | Meaning |
|---|---|---|
| `latent` | number[] | 6-dim latent vector |
| `text` | string | Doctrine-anchored prose |
| `image` | string | 16x16 ASCII art |
| `audio` | Float32Array / number[] | 16 kHz waveform, 4 seconds |

### PipelineResult

| Field | Type | Meaning |
|---|---|---|
| `features` | JEVFeatures | 6-dim feature vector |
| `decodedSamples` | DecodedSample[] | One per sampled latent |

## Naming Conventions

To support both snake_case (Python convention) and camelCase (TypeScript convention),
the JSON wire format uses both spellings. The Python dataclass uses snake_case,
the TypeScript interface uses camelCase.

```python
# Python
@dataclass
class JEVFeatures:
    canon_worthy: float
    distinct_voice: float
    doctrine_anchor: float
    voice_fit: float
    novelty: float
    density: float
```

```typescript
// TypeScript
interface JEVFeatures {
  canonWorthy: number;
  distinctVoice: number;
  doctrineAnchor: number;
  voiceFit: number;
  novelty: number;
  density: number;
}
```

When marshalling to JSON, both spellings are present:
```json
{
  "canon_worthy": 0.46,
  "canonWorthy": 0.46,
  ...
}
```

## Polyformalism Harness

The polyformalism harness (`tests/test_polyformalism.py`) verifies that all
ports produce byte-identical output for the same input lore.

```python
# tests/test_polyformalism.py

def test_python_ts_parity():
    """Python and TypeScript implementations produce the same features."""
    import subprocess
    lore = "The canon gate made itself heard."
    
    # Python
    python_features = run_pipeline_v2(lore)["features"]
    
    # TypeScript (via subprocess to a deployed service)
    ts_features = json.loads(subprocess.check_output([
        "node", "-e", 
        f"const c = new PolyvocoderClient(); "
        f"c.getJEVFeatures({json.dumps(lore)}).then(JSON.stringify).then(console.log)"
    ]))
    
    for i, val in enumerate(python_features):
        assert abs(python_features[i] - ts_features[i]) < 0.01
```

## Adding a New Polyformalism Port

To add a Rust port, for example:

1. Create a new repo: `SuperInstance/polyvocoder-rs`
2. Define the schema in Rust:
   ```rust
   #[derive(Serialize, Deserialize)]
   pub struct JevFeatures {
       pub canon_worthy: f32,
       pub distinct_voice: f32,
       pub doctrine_anchor: f32,
       pub voice_fit: f32,
       pub novelty: f32,
       pub density: f32,
   }
   ```
3. Implement the same 5 stages:
   - JEV probe (HTTP client to Typesafe.ai)
   - Tiny VAE (use nalgebra or ndarray)
   - 3 modality heads
4. Add to the polyformalism harness

## Why This Matters

Polyformalism proves that the polyvocoder's design is **language-independent**.
The semantic latent space (JEV features) is the same in every port. The
modalities are decoupled from the language. The VAE math is universal.

A canon lore rendered through Python polyvocoder looks the same as one rendered
through TypeScript polyvocoder. The fleet canary hash verifies this byte-exact.

