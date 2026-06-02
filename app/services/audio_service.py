"""
Audio extraction and processing service
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple
import soundfile as sf
import numpy as np
import logging

# Lazy import for librosa and matplotlib 
try:
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    SPECTROGRAM_AVAILABLE = True
except ImportError:
    SPECTROGRAM_AVAILABLE = False
    logging.warning("librosa or matplotlib not installed. Spectrogram generation unavailable.")

from app.config import settings

logger = logging.getLogger(__name__)


class AudioService:
    """Service for extracting and caching audio snippets"""

    def __init__(self):
        # Use /tmp for cache (writable), not /data (read-only)
        self.cache_dir = Path("/tmp/snippet_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_snippet(
        self,
        recording_file_path: str,
        start_time: float,
        end_time: float,
        snippet_id: int,
        use_cache: bool = True,
    ) -> Tuple[Path, int, int]:
        """
        Extract audio snippet from recording.

        Args:
            recording_file_path: Path to the original recording file
            start_time: Start time in seconds
            end_time: End time in seconds
            snippet_id: ID of the snippet (for caching)
            use_cache: Whether to use cached file if available

        Returns:
            Tuple of (audio_file_path, sample_rate, channels)

        Raises:
            FileNotFoundError: If recording file doesn't exist
            ValueError: If time range is invalid
        """
        # Validate time range
        if start_time < 0 or end_time <= start_time:
            raise ValueError(f"Invalid time range: {start_time} - {end_time}")

        # Check cache first
        if use_cache:
            cached_path = self._get_cached_path(snippet_id)
            if cached_path.exists():
                logger.debug(f"Serving cached snippet {snippet_id}")
                info = sf.info(str(cached_path))
                return cached_path, info.samplerate, info.channels

        # Resolve full path
        DATA_ROOT = settings.DATA_ROOT or "/data"
        if not recording_file_path.startswith("/"):
            full_path = os.path.join(DATA_ROOT, recording_file_path)
        else:
            full_path = recording_file_path

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Recording file not found: {full_path}")

        # Extract snippet
        try:
            info = sf.info(full_path)
            sample_rate = info.samplerate

            # Calculate frame positions
            start_frame = int(start_time * sample_rate)
            end_frame = int(end_time * sample_rate)
            frames_to_read = end_frame - start_frame

            # Ensure we don't exceed file bounds
            if start_frame >= info.frames:
                raise ValueError(f"Start time {start_time}s exceeds recording duration")

            if end_frame > info.frames:
                logger.warning(
                    f"End time {end_time}s exceeds recording duration, "
                    f"truncating to {info.frames / sample_rate}s"
                )
                frames_to_read = info.frames - start_frame

            # Read audio segment
            audio_data, _ = sf.read(
                full_path,
                start=start_frame,
                frames=frames_to_read,
                dtype='float32'
            )
            # Handle stereo by converting to mono
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            # Save to cache
            cached_path = self._get_cached_path(snippet_id)
            sf.write(
                str(cached_path),
                audio_data,
                sample_rate,
                format='WAV',
                subtype='PCM_16'
            )

            logger.info(f"Extracted and cached snippet {snippet_id}")
            return cached_path, sample_rate, info.channels

        except Exception as e:
            logger.error(f"Failed to extract snippet {snippet_id}: {e}", exc_info=True)
            raise

    def _get_cached_path(self, snippet_id: int) -> Path:
        """Get cache file path for a snippet"""
        # Organize cache into subdirectories (1000 snippets per dir)
        subdir = self.cache_dir / f"{snippet_id // 1000}"
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"snippet_{snippet_id}.wav"

    def get_cached_path_if_exists(self, snippet_id: int) -> Optional[Path]:
        """Check if cached snippet exists and return path"""
        cached_path = self._get_cached_path(snippet_id)
        return cached_path if cached_path.exists() else None

    def clear_cache(self, snippet_id: Optional[int] = None):
        """
        Clear cached snippets.
        
        Args:
            snippet_id: If provided, clear only this snippet. Otherwise clear all.
        """
        if snippet_id is not None:
            cached_path = self._get_cached_path(snippet_id)
            if cached_path.exists():
                cached_path.unlink()
                logger.info(f"Cleared cache for snippet {snippet_id}")
        else:
            # Clear entire cache directory
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared entire snippet cache")

    def get_cache_stats(self) -> dict:
        """Get statistics about cached snippets"""
        if not self.cache_dir.exists():
            return {"cached_count": 0, "total_size_mb": 0}

        cached_files = list(self.cache_dir.rglob("snippet_*.wav"))
        total_size = sum(f.stat().st_size for f in cached_files)

        return {
            "cached_count": len(cached_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }

    def generate_spectrogram(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        snippet_id: int,
        use_cache: bool = True,
        n_mels: int = 128,
        fmin: float = 0,
        fmax: Optional[float] = None,
    ) -> Path:
        """
        Generate mel spectrogram visualization for audio snippet.

        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            snippet_id: ID of the snippet (for caching)
            use_cache: Whether to use cached spectrogram if available
            n_mels: Number of mel bands (default: 128)
            fmin: Minimum frequency (Hz)
            fmax: Maximum frequency (Hz), defaults to sr/2

        Returns:
            Path to the generated spectrogram PNG file

        Raises:
            ImportError: If librosa or matplotlib not installed
            Exception: If spectrogram generation fails
        """
        if not SPECTROGRAM_AVAILABLE:
            raise ImportError(
                "librosa and matplotlib are required for spectrogram generation. "
                "Install with: pip install librosa matplotlib"
            )

        # Check cache first
        if use_cache:
            cached_path = self._get_spectrogram_cache_path(snippet_id)
            if cached_path.exists():
                logger.debug(f"Serving cached spectrogram for snippet {snippet_id}")
                return cached_path

        try:
            # Set default fmax if not provided
            if fmax is None:
                fmax = sample_rate / 2

            # Generate mel spectrogram
            S = librosa.feature.melspectrogram(
                y=audio_data,
                sr=sample_rate,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
                n_fft=2048,
                hop_length=512,
            )
            
            # Convert to dB scale
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Create figure with appropriate size
            fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
            
            # Plot spectrogram
            img = librosa.display.specshow(
                S_dB,
                x_axis='time',
                y_axis='mel',
                sr=sample_rate,
                fmin=fmin,
                fmax=fmax,
                ax=ax,
                cmap='viridis'
            )
            
            # Add colorbar
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            
            # Set labels
            ax.set(
                title=f'Mel Spectrogram - Snippet {snippet_id}',
                xlabel='Time (s)',
                ylabel='Frequency (Hz)'
            )
            
            # Save to cache
            cached_path = self._get_spectrogram_cache_path(snippet_id)
            plt.savefig(
                cached_path,
                dpi=100,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            plt.close(fig)

            logger.info(f"Generated and cached spectrogram for snippet {snippet_id}")
            return cached_path

        except Exception as e:
            logger.error(f"Failed to generate spectrogram for snippet {snippet_id}: {e}", exc_info=True)
            # Clean up matplotlib figure if it exists
            plt.close('all')
            raise

    def _get_spectrogram_cache_path(self, snippet_id: int) -> Path:
        """Get cache file path for a spectrogram"""
        # Organize cache into subdirectories (1000 spectrograms per dir)
        subdir = self.cache_dir / f"spectrograms" / f"{snippet_id // 1000}"
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"spectrogram_{snippet_id}.png"

    def get_spectrogram_if_cached(self, snippet_id: int) -> Optional[Path]:
        """Check if cached spectrogram exists and return path"""
        cached_path = self._get_spectrogram_cache_path(snippet_id)
        return cached_path if cached_path.exists() else None


# Global instance
audio_service = AudioService()

