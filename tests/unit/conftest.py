import pytest
from app.database import Base

@pytest.fixture(autouse=True)
def reset_db(engine):
    # Drop everything
    Base.metadata.drop_all(bind=engine)
    # Recreate fresh tables
    Base.metadata.create_all(bind=engine)
    yield
