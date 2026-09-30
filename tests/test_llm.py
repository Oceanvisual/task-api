from unittest.mock import patch

from app import llm


@patch("app.llm.ChatOpenAI")
def test_get_llm_no_proxy_by_default(mock_chat) -> None:
    with patch.object(llm.settings, "llm_proxy_url", ""):
        llm.get_llm()
    kwargs = mock_chat.call_args.kwargs
    assert "http_client" not in kwargs
    assert "http_async_client" not in kwargs


@patch("app.llm.DefaultAsyncHttpxClient")
@patch("app.llm.DefaultHttpxClient")
@patch("app.llm.ChatOpenAI")
def test_get_llm_wires_proxy_when_set(mock_chat, mock_sync, mock_async) -> None:
    proxy = "http://user:pass@gate.decodo.com:7000"
    with patch.object(llm.settings, "llm_proxy_url", proxy):
        llm.get_llm()

    # прокси реально прокинут в оба http-клиента openai SDK
    mock_sync.assert_called_once_with(proxy=proxy)
    mock_async.assert_called_once_with(proxy=proxy)
    kwargs = mock_chat.call_args.kwargs
    assert kwargs["http_client"] is mock_sync.return_value
    assert kwargs["http_async_client"] is mock_async.return_value
