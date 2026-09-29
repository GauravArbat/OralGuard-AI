"""
OralGuard AI — Database Connection & Session Management
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from config import settings


# ── Async Engine ──
# For SQLite dev mode, use aiosqlite driver
if settings.DATABASE_URL.startswith("sqlite"):
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        connect_args={"check_same_thread": False},
    )
else:
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True,
    )

# ── Session Factory ──
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── Base Model ──
class Base(DeclarativeBase):
    pass


# ── Dependency ──
async def get_db() -> AsyncSession:
    """FastAPI dependency that yields a database session."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Init DB ──
async def init_db():
    """Create all tables on startup and seed default user."""
    from sqlalchemy import select
    from models.db_models import (
        User, UserRole, Screening, ScreeningImage,
        QuestionnaireResponse, DiagnosisReport
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Ensure anonymous user exists for unauthenticated screenings
    async with async_session() as session:
        result = await session.execute(select(User).where(User.id == "anonymous"))
        if not result.scalar_one_or_none():
            anon = User(
                id="anonymous",
                full_name="Anonymous Patient",
                role=UserRole.PATIENT,
                is_active=True,
            )
            session.add(anon)
            await session.commit()

