"""
Database Configuration and Session Management
Uses SQLAlchemy ORM and DATABASE_URL from environment variable.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Default database URL
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    try:
        import psycopg2  # noqa: F401 — verify postgres driver installed
        DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/dsp_app"
    except ImportError:
        DATABASE_URL = "sqlite:///./dsp_app.db"

# Fix for potential postgres:// vs postgresql:// dialect issue in some connection strings
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Handle sqlite specific connect args if fallback used
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

try:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
except Exception:
    # If postgresql connection dialect fails due to missing dbapi driver, fallback to sqlite
    DATABASE_URL = "sqlite:///./dsp_app.db"
    connect_args = {"check_same_thread": False}
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        connect_args=connect_args,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI Dependency for database session lifecycle."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initializes tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)
