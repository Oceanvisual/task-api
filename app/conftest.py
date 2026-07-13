import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from main import app, tasks


@pytest_asyncio.fixture
async def client():
    # очищаем хранилище перед каждым тестом
    tasks.clear()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c