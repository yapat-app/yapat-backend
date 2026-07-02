from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer


class BirdNetEmbedder:
    """
    BirdNET V2.4 embedder (raw waveform → 1024-d embedding).

    - Uses SavedModel endpoint "embeddings".
    - Expects exactly 144000 float32 samples (3s @ 48 kHz).
    - Returns Python list[float].
    """

    _instance = None
    _gpu_memory_growth_configured = False
    _model_path = (
            Path(__file__).resolve().parent
            / ".." / "assets" / "models" / "birdnet"
    )

    SAMPLE_RATE = 48000
    WINDOW_SAMPLES = 144000  # 3 seconds

    @classmethod
    def _configure_gpu_memory_growth(cls):
        """
        Without this, TensorFlow claims nearly all memory on the first visible
        GPU as soon as it runs its first op — for every process that touches
        it. Celery's prefork pool (--concurrency) forks multiple worker
        processes, each independently loading this model; the first one to
        initialize can grab the entire GPU, starving sibling processes of
        memory for cuFFT scratch buffers ("Failed to create cuFFT batched
        plan"). Memory growth makes each process allocate incrementally
        instead, so multiple worker processes can actually share one GPU.
        Must run before any GPU op executes, so it's called first thing in
        instance(), not lazily at inference time.
        """
        if cls._gpu_memory_growth_configured:
            return
        cls._gpu_memory_growth_configured = True
        try:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Only settable before GPUs are initialized; if something already
            # triggered initialization first, there's nothing more to do here.
            pass

    @classmethod
    def instance(cls):
        """Load BirdNET SavedModel endpoint once per worker."""
        if cls._instance is None:
            cls._configure_gpu_memory_growth()
            cls._instance = TFSMLayer(
                str(cls._model_path),
                call_endpoint="embeddings"  # Your endpoint name
            )
        return cls._instance

    @classmethod
    def embed(cls, audio_path: str, start_time: float):
        """
        Extract a BirdNET embedding for a 3-second snippet.

        Args:
            audio_path: path to audio file
            start_time: snippet start time in seconds

        Returns:
            list[float] (size 1024) or None
        """

        # 1. Load exactly 3 seconds from file
        audio, _ = librosa.load(
            audio_path,
            sr=cls.SAMPLE_RATE,
            offset=start_time,
            duration=3.0,
            mono=True
        )

        if audio.size == 0:
            return None

        # 2. Pad or trim to match model requirements
        # TODO Assert file has the expected length
        if len(audio) < cls.WINDOW_SAMPLES:
            audio = np.pad(audio, (0, cls.WINDOW_SAMPLES - len(audio)))
        else:
            audio = audio[:cls.WINDOW_SAMPLES]

        # Shape: (1, 144000)
        batch = audio.astype(np.float32)[None, :]

        # 3. Run inference
        model = cls.instance()
        outputs = model(batch)

        # 4. Extract embedding
        emb = outputs["embeddings"].numpy()[0]

        return emb.tolist()

    @classmethod
    def embed_batch_from_recording(cls, audio_path: str, snippet_windows: list[tuple[float, float]]):
        """
        Extract BirdNET embeddings for multiple snippets from the same recording.
        
        Much more efficient than calling embed() repeatedly:
        - Loads the full audio file once
        - Runs batch inference on all snippets together
        - Reduces TensorFlow invocation overhead
        
        Args:
            audio_path: path to audio file
            snippet_windows: list of (start_time, end_time) tuples in seconds
            
        Returns:
            list[list[float]] - one embedding per snippet (same order as input)
            Returns None for snippets that fail to extract
        """
        if not snippet_windows:
            return []
        
        # 1. Load the entire audio file once
        full_audio, _ = librosa.load(
            audio_path,
            sr=cls.SAMPLE_RATE,
            mono=True
        )
        
        if full_audio.size == 0:
            return [None] * len(snippet_windows)
        
        # 2. Extract all snippet windows from the loaded audio
        batch_samples = []
        valid_indices = []
        
        for idx, (start_time, end_time) in enumerate(snippet_windows):
            start_sample = int(start_time * cls.SAMPLE_RATE)
            end_sample = int(end_time * cls.SAMPLE_RATE)
            
            # Extract the window
            snippet_audio = full_audio[start_sample:end_sample]
            
            # Skip if no audio data
            if snippet_audio.size == 0:
                continue
            
            # Pad or trim to exactly WINDOW_SAMPLES
            if len(snippet_audio) < cls.WINDOW_SAMPLES:
                snippet_audio = np.pad(snippet_audio, (0, cls.WINDOW_SAMPLES - len(snippet_audio)))
            else:
                snippet_audio = snippet_audio[:cls.WINDOW_SAMPLES]
            
            batch_samples.append(snippet_audio)
            valid_indices.append(idx)
        
        # 3. If no valid snippets, return all None
        if not batch_samples:
            return [None] * len(snippet_windows)
        
        # 4. Stack into batch tensor: (N, 144000)
        batch = np.stack(batch_samples, axis=0).astype(np.float32)
        
        # 5. Run batch inference
        model = cls.instance()
        outputs = model(batch)
        
        # 6. Extract embeddings
        embeddings = outputs["embeddings"].numpy()
        
        # 7. Map results back to original order (insert None for failed snippets)
        results = [None] * len(snippet_windows)
        for batch_idx, original_idx in enumerate(valid_indices):
            results[original_idx] = embeddings[batch_idx].tolist()
        
        return results
