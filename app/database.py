from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings


def get_database_url() -> str:
    """
    Get database URL, converting PostgreSQL URLs to use psycopg3 explicitly.
    
    SQLAlchemy 2.0.44 defaults to psycopg2 for postgresql:// URLs.
    Since we're using psycopg3 (psycopg), we need to explicitly specify it.
    
    Railway provides postgresql:// or postgres:// URLs, but SQLAlchemy will
    try to use psycopg2 (not installed). We convert to postgresql+psycopg://
    or postgres+psycopg:// to force SQLAlchemy to use psycopg3.
    """
    url = settings.DATABASE_URL
    
    # Handle postgresql:// format (most common from Railway)
    if url.startswith("postgresql://") and not url.startswith("postgresql+psycopg://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    # Handle postgres:// format (shorter alias, also used by some providers)
    elif url.startswith("postgres://") and not url.startswith("postgres+psycopg://"):
        url = url.replace("postgres://", "postgres+psycopg://", 1)
    
    return url


engine = create_engine(
    get_database_url(),
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
