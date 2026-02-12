"""
WSSED species extraction utility

Extracts species names from FNJV dataset file naming convention.
Format: FNJV_[ID]_[Genus]_[species]_[Location]_[Collector].wav
"""

from typing import List, Set
from sqlalchemy.orm import Session
import logging

from app.models.recording import Recording

logger = logging.getLogger(__name__)


def extract_species_from_filename(filename: str) -> str:
    """
    Extract species name from FNJV filename format.
    
    Args:
        filename: Audio filename (e.g., 'FNJV_0001580_Dendropsophus_minutus_Nova_Friburgo_RJ_Adao_Jose_Cardoso.wav')
    
    Returns:
        Species name in format 'Genus_species' (e.g., 'Dendropsophus_minutus')
    
    Raises:
        ValueError: If filename doesn't match expected format
    
    Examples:
        >>> extract_species_from_filename('FNJV_0001580_Dendropsophus_minutus_Nova_Friburgo_RJ.wav')
        'Dendropsophus_minutus'
    """
    # Remove .wav extension
    base = filename.replace('.wav', '').replace('.WAV', '')
    
    # Split by underscore
    parts = base.split('_')
    
    # Validate format (minimum: FNJV_ID_Genus_species)
    if len(parts) < 4:
        raise ValueError(
            f"Invalid filename format: {filename}. "
            f"Expected format: FNJV_[ID]_[Genus]_[species]_[...]"
        )
    
    # Extract genus (part 2) and species (part 3)
    genus = parts[2]
    species = parts[3]
    
    # Validate that genus and species are not empty
    if not genus or not species:
        raise ValueError(f"Invalid species name in filename: {filename}")
    
    return f"{genus}_{species}"


def get_dataset_species_list(dataset_id: int, db: Session) -> List[str]:
    """
    Get unique species list from all recordings in a dataset.
    
    Args:
        dataset_id: ID of the dataset
        db: Database session
    
    Returns:
        Sorted list of unique species names
    
    Examples:
        >>> get_dataset_species_list(1, db)
        ['Dendropsophus_minutus', 'Boana_raniceps', ...]
    """
    recordings = db.query(Recording).filter(
        Recording.dataset_id == dataset_id
    ).all()
    
    species_set: Set[str] = set()
    invalid_files: List[str] = []
    
    for rec in recordings:
        try:
            species = extract_species_from_filename(rec.file_name)
            species_set.add(species)
        except ValueError as e:
            logger.warning(f"Could not extract species from {rec.file_name}: {e}")
            invalid_files.append(rec.file_name)
            continue
    
    if invalid_files:
        logger.info(
            f"Skipped {len(invalid_files)} files with invalid naming format in dataset {dataset_id}"
        )
    
    return sorted(list(species_set))


def get_species_distribution(dataset_id: int, db: Session) -> dict:
    """
    Get distribution of species across recordings in a dataset.
    
    Args:
        dataset_id: ID of the dataset
        db: Database session
    
    Returns:
        Dictionary mapping species names to recording counts
    
    Examples:
        >>> get_species_distribution(1, db)
        {'Dendropsophus_minutus': 45, 'Boana_raniceps': 32, ...}
    """
    recordings = db.query(Recording).filter(
        Recording.dataset_id == dataset_id
    ).all()
    
    distribution = {}
    
    for rec in recordings:
        try:
            species = extract_species_from_filename(rec.file_name)
            distribution[species] = distribution.get(species, 0) + 1
        except ValueError:
            continue
    
    return distribution


def format_species_for_display(species_name: str) -> str:
    """
    Format species name for human-readable display.
    
    Args:
        species_name: Species in 'Genus_species' format
    
    Returns:
        Species in 'Genus species' format (space instead of underscore)
    
    Examples:
        >>> format_species_for_display('Dendropsophus_minutus')
        'Dendropsophus minutus'
    """
    return species_name.replace('_', ' ')


def get_genus_from_species(species_name: str) -> str:
    """
    Extract genus from species name.
    
    Args:
        species_name: Species in 'Genus_species' format
    
    Returns:
        Genus name
    
    Examples:
        >>> get_genus_from_species('Dendropsophus_minutus')
        'Dendropsophus'
    """
    return species_name.split('_')[0]


def validate_species_name(species_name: str) -> bool:
    """
    Validate that a species name follows the expected format.
    
    Args:
        species_name: Species name to validate
    
    Returns:
        True if valid, False otherwise
    
    Examples:
        >>> validate_species_name('Dendropsophus_minutus')
        True
        >>> validate_species_name('invalid')
        False
    """
    parts = species_name.split('_')
    return len(parts) == 2 and all(part.strip() for part in parts)
