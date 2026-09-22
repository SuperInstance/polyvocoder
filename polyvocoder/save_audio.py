"""Save polyvocoder audio output as WAV file."""
import sys
import wave
import numpy as np
from pathlib import Path

def save_wav(audio: np.ndarray, path: str, sample_rate: int = 16000):
    """Save float32 audio array as 16-bit PCM WAV."""
    # Convert float32 [-1, 1] to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return path


if __name__ == "__main__":
    sys.path.insert(0, '/workspace/repos/polyvocoder')
    from polyvocoder.pipeline_v2 import run_pipeline_v2

    lore = """The canon gate made itself heard.
Not opened. Not breached. Heard.
The cells are scars. That's the whole scripture."""

    result = run_pipeline_v2(lore, n_samples=3)
    out_dir = Path("/workspace/repos/polyvocoder/outputs")
    out_dir.mkdir(exist_ok=True)

    for i, sample in enumerate(result['decoded_samples']):
        # Save audio
        wav_path = out_dir / f"sample_{i}.wav"
        save_wav(sample['audio'], str(wav_path))
        print(f"Saved {wav_path}")
        # Save text
        text_path = out_dir / f"sample_{i}.txt"
        text_path.write_text(sample['text'])
        print(f"Saved {text_path}")
        # Save image
        image_path = out_dir / f"sample_{i}.txt"
        with open(image_path, 'a') as f:
            f.write("\n\n=== IMAGE ===\n")
            f.write(sample['image'])
        print(f"Saved {image_path}")
