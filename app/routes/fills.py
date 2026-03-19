from fastapi import APIRouter
from sqlalchemy import select

from app.db import Fill, async_session

router = APIRouter(prefix="/api")


@router.get("/fills")
async def list_fills(ticker: str | None = None, limit: int = 100):
    """Return filled trades from local DB."""
    async with async_session() as session:
        q = select(Fill).order_by(Fill.filled_at.desc()).limit(limit)
        if ticker:
            q = q.where(Fill.ticker == ticker)
        result = await session.execute(q)
        rows = result.scalars().all()
    return {
        "fills": [
            {
                "id": f.id,
                "order_id": f.order_id,
                "ticker": f.ticker,
                "side": f.side,
                "price": f.price,
                "count": f.count,
                "filled_at": f.filled_at.isoformat() if f.filled_at else None,
            }
            for f in rows
        ]
    }
