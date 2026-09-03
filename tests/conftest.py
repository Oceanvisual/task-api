from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Settings требует LLM_API_KEY при импорте app.config → app.main
# В CI ключ может отсутствовать; для unit-тестов достаточно dummy.
import os

os.environ.setdefault("LLM_API_KEY", "test-key-not-used")


@pytest.fixture
def client():
    mock_chain = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    with patch("app.main.build_rag_chain", return_value=(mock_chain, mock_retriever)):
        from app.main import app

        with TestClient(app) as c:
            yield c
