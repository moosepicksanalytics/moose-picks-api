"""
Database migration: Add closing_total, actual_total, and ou_result columns to games table.
Run this once to add the new columns to existing database.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.database import engine
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_add_ou_columns():
    """Add O/U columns to games table if they don't exist"""
    
    with engine.connect() as conn:
        try:
            # Check if columns already exist
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'games' 
                AND column_name IN ('closing_total', 'actual_total', 'ou_result')
            """))
            existing_columns = [row[0] for row in result]
            
            # Add columns that don't exist
            if 'closing_total' not in existing_columns:
                logger.info("Adding closing_total column...")
                conn.execute(text("ALTER TABLE games ADD COLUMN closing_total FLOAT"))
                conn.commit()
                logger.info("✓ Added closing_total column")
            else:
                logger.info("closing_total column already exists")
            
            if 'actual_total' not in existing_columns:
                logger.info("Adding actual_total column...")
                conn.execute(text("ALTER TABLE games ADD COLUMN actual_total INTEGER"))
                conn.commit()
                logger.info("✓ Added actual_total column")
            else:
                logger.info("actual_total column already exists")
            
            if 'ou_result' not in existing_columns:
                logger.info("Adding ou_result column...")
                conn.execute(text("ALTER TABLE games ADD COLUMN ou_result VARCHAR(10)"))
                conn.commit()
                logger.info("✓ Added ou_result column")
            else:
                logger.info("ou_result column already exists")
            
            # Create index for faster queries
            try:
                logger.info("Creating index on ou_result...")
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_games_ou_result ON games(sport, ou_result)"))
                conn.commit()
                logger.info("✓ Created index on ou_result")
            except Exception as e:
                logger.warning(f"Index creation failed (may already exist): {e}")
            
            logger.info("\n✓ Migration complete! All O/U columns added.")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            import traceback
            traceback.print_exc()
            conn.rollback()
            raise


if __name__ == "__main__":
    migrate_add_ou_columns()
