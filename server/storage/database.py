"""
Database Layer
--------------
SQLAlchemy async ORM for PostgreSQL.
Tables: defects, objects, cameras
"""

from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy import String, Float, Boolean, DateTime, Text, Integer
from loguru import logger


class Base(DeclarativeBase):
    pass


class DefectRecord(Base):
    __tablename__ = "defects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    global_id: Mapped[str] = mapped_column(String(16), index=True)
    camera_id: Mapped[str] = mapped_column(String(32))
    label: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    bbox_x1: Mapped[float] = mapped_column(Float)
    bbox_y1: Mapped[float] = mapped_column(Float)
    bbox_x2: Mapped[float] = mapped_column(Float)
    bbox_y2: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(32))  # DEFECT_CONFIRMED | UNCERTAIN
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Database:
    def __init__(self, url: str):
        self.engine = create_async_engine(url, echo=False, pool_size=10)
        self.session_factory = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def connect(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.success("Database connected and schema applied.")

    async def disconnect(self):
        await self.engine.dispose()

    async def insert_defect(
        self,
        global_id: str,
        camera_id: str,
        label: str,
        confidence: float,
        bbox: list[float],
        decision: str,
        timestamp: float,
    ):
        record = DefectRecord(
            global_id=global_id,
            camera_id=camera_id,
            label=label,
            confidence=confidence,
            bbox_x1=bbox[0],
            bbox_y1=bbox[1],
            bbox_x2=bbox[2],
            bbox_y2=bbox[3],
            decision=decision,
            timestamp=datetime.utcfromtimestamp(timestamp),
        )
        async with self.session_factory() as session:
            session.add(record)
            await session.commit()

    async def get_recent_defects(self, limit: int = 100) -> list[DefectRecord]:
        from sqlalchemy import select, desc
        async with self.session_factory() as session:
            result = await session.execute(
                select(DefectRecord)
                .order_by(desc(DefectRecord.created_at))
                .limit(limit)
            )
            return result.scalars().all()
