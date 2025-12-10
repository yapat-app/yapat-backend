"""
Mock data management script for testing annotation workflow

Commands:
  python -m scripts.mock_data create   # Create mock data
  python -m scripts.mock_data clear    # Remove mock data
  python -m scripts.mock_data status   # Show mock data status

This script creates a test dataset with recordings and snippets for
annotation workflow testing. Mock data is tagged for easy identification
and removal.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.dataset import Dataset
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.annotation import Annotation

# Constants
MOCK_DATASET_NAME = "Mock Bird Dataset"
MOCK_MARKER = "MOCK_DATA"


def create_mock_data() -> None:
    """Create mock recordings and snippets for testing"""
    db = SessionLocal()
    try:
        print("=" * 60)
        print("Creating Mock Data for Annotation Workflow")
        print("=" * 60)
        
        # Check if mock data already exists
        existing_dataset = db.query(Dataset).filter(
            Dataset.name == MOCK_DATASET_NAME
        ).first()
        
        if existing_dataset:
            print(f"\n⚠️  Mock dataset already exists (ID: {existing_dataset.id})")
            print("   Run 'clear' command first to recreate.")
            return
        
        # Create dataset
        dataset = Dataset(
            name=MOCK_DATASET_NAME,
            description=f"[{MOCK_MARKER}] Test dataset with mock bird recordings for annotation workflow testing",
            source_uri="/mock/audio/files",
            team_id=None
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        print(f"\n✓ Created dataset: {dataset.name}")
        print(f"  ID: {dataset.id}")
        
        # Create recordings with varying durations
        recordings_data = [
            {
                "name": "morning_chorus_001.wav",
                "duration": 300.0,
                "description": "Morning bird chorus recording"
            },
            {
                "name": "evening_calls_002.wav",
                "duration": 180.0,
                "description": "Evening bird calls"
            },
            {
                "name": "dawn_songs_003.wav",
                "duration": 240.0,
                "description": "Dawn bird songs"
            },
            {
                "name": "wetland_sounds_004.wav",
                "duration": 360.0,
                "description": "Wetland bird sounds"
            },
        ]
        
        recordings = []
        for rec_data in recordings_data:
            recording = Recording(
                dataset_id=dataset.id,
                file_path=f"/mock/audio/{rec_data['name']}",
                file_name=rec_data["name"],
                duration=rec_data["duration"],
                sample_rate=48000
            )
            db.add(recording)
            recordings.append(recording)
        
        db.commit()
        
        for rec in recordings:
            db.refresh(rec)
        
        print(f"\n✓ Created {len(recordings)} recordings:")
        for rec in recordings:
            print(f"  - {rec.file_name} ({rec.duration}s)")
        
        # Create snippets using sliding window
        window_duration = 3.0  # 3-second windows
        hop_duration = 1.5     # 1.5-second hop (50% overlap)
        total_snippets = 0
        
        print(f"\n✓ Generating snippets (window={window_duration}s, hop={hop_duration}s)...")
        
        for recording in recordings:
            start_time = 0.0
            recording_snippets = []
            
            while start_time + window_duration <= recording.duration:
                snippet = Snippet(
                    recording_id=recording.id,
                    start_time=start_time,
                    end_time=start_time + window_duration,
                    duration=window_duration,
                    file_path=None,  # Will be set when audio is actually extracted
                    is_annotated=False
                )
                recording_snippets.append(snippet)
                start_time += hop_duration
            
            db.add_all(recording_snippets)
            total_snippets += len(recording_snippets)
            print(f"  - {recording.file_name}: {len(recording_snippets)} snippets")
        
        db.commit()
        
        # Summary
        print("\n" + "=" * 60)
        print("Mock Data Created Successfully!")
        print("=" * 60)
        print(f"Dataset ID:        {dataset.id}")
        print(f"Dataset Name:      {dataset.name}")
        print(f"Total Recordings:  {len(recordings)}")
        print(f"Total Snippets:    {total_snippets}")
        print(f"\nTo view status:    python -m scripts.mock_data status")
        print(f"To remove:         python -m scripts.mock_data clear")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error creating mock data: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def clear_mock_data() -> None:
    """Remove all mock data"""
    db = SessionLocal()
    try:
        print("=" * 60)
        print("Clearing Mock Data")
        print("=" * 60)
        
        dataset = db.query(Dataset).filter(Dataset.name == MOCK_DATASET_NAME).first()
        
        if not dataset:
            print("\n✓ No mock data found. Nothing to clear.")
            return
        
        # Get counts before deletion
        recordings = db.query(Recording).filter(
            Recording.dataset_id == dataset.id
        ).all()
        
        snippets_count = db.query(Snippet).join(Recording).filter(
            Recording.dataset_id == dataset.id
        ).count()
        
        annotations_count = db.query(Annotation).join(Snippet).join(Recording).filter(
            Recording.dataset_id == dataset.id
        ).count()
        
        print(f"\nFound mock data:")
        print(f"  Dataset:     {dataset.name} (ID: {dataset.id})")
        print(f"  Recordings:  {len(recordings)}")
        print(f"  Snippets:    {snippets_count}")
        print(f"  Annotations: {annotations_count}")
        
        # Confirm deletion
        print(f"\nDeleting dataset (cascades to all related data)...")
        
        # Delete dataset (cascades due to ondelete='CASCADE')
        db.delete(dataset)
        db.commit()
        
        print("\n" + "=" * 60)
        print("Mock Data Cleared Successfully!")
        print("=" * 60)
        print(f"Removed:")
        print(f"  - 1 dataset")
        print(f"  - {len(recordings)} recordings")
        print(f"  - {snippets_count} snippets")
        print(f"  - {annotations_count} annotations")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error clearing mock data: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def show_status() -> None:
    """Show mock data status"""
    db = SessionLocal()
    try:
        print("=" * 60)
        print("Mock Data Status")
        print("=" * 60)
        
        dataset = db.query(Dataset).filter(Dataset.name == MOCK_DATASET_NAME).first()
        
        if not dataset:
            print("\n✗ No mock data found.")
            print("\nTo create mock data:")
            print("  python -m scripts.mock_data create")
            return
        
        recordings = db.query(Recording).filter(
            Recording.dataset_id == dataset.id
        ).all()
        
        snippets = db.query(Snippet).join(Recording).filter(
            Recording.dataset_id == dataset.id
        ).all()
        
        annotated_snippets = [s for s in snippets if s.is_annotated]
        
        annotations_count = db.query(Annotation).join(Snippet).join(Recording).filter(
            Recording.dataset_id == dataset.id
        ).count()
        
        print(f"\n✓ Mock data exists:")
        print(f"\n  Dataset: {dataset.name}")
        print(f"  ID:      {dataset.id}")
        print(f"\n  Recordings:         {len(recordings)}")
        print(f"  Total Snippets:     {len(snippets)}")
        print(f"  Annotated Snippets: {len(annotated_snippets)}")
        print(f"  Unannotated:        {len(snippets) - len(annotated_snippets)}")
        print(f"  Total Annotations:  {annotations_count}")
        
        if recordings:
            print(f"\n  Recordings breakdown:")
            for rec in recordings:
                rec_snippets = [s for s in snippets if s.recording_id == rec.id]
                print(f"    - {rec.file_name}: {len(rec_snippets)} snippets")
        
        print("\n" + "=" * 60)
        
    finally:
        db.close()


def main():
    """Main entry point"""
    commands = {
        "create": create_mock_data,
        "clear": clear_mock_data,
        "status": show_status,
    }
    
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage: python -m scripts.mock_data {create|clear|status}")
        print("\nCommands:")
        print("  create  - Create mock dataset with recordings and snippets")
        print("  clear   - Remove all mock data")
        print("  status  - Show current mock data status")
        sys.exit(1)
    
    commands[sys.argv[1]]()


if __name__ == "__main__":
    main()

