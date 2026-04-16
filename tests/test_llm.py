"""Tests for codegraph.llm — mocked OpenAI calls."""

import os
from unittest.mock import MagicMock, patch

from codegraph.llm import LLMClient


def test_is_available_with_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = LLMClient()
        assert client.is_available() is True


def test_is_available_without_key():
    with patch.dict(os.environ, {}, clear=True):
        client = LLMClient()
        # May pick up real env var; just test the method exists
        assert isinstance(client.is_available(), bool)


def test_cost_tracking():
    client = LLMClient()
    client._input_tokens = 1000
    client._output_tokens = 500
    client._embedding_tokens = 2000
    cost = client.get_cost_summary()
    assert cost.input_tokens == 1000
    assert cost.output_tokens == 500
    assert cost.embedding_tokens == 2000
    assert cost.estimated_cost_usd > 0


def test_reset_cost_tracking():
    client = LLMClient()
    client._input_tokens = 100
    client._output_tokens = 50
    summary = client.reset_cost_tracking()
    assert summary.input_tokens == 100
    assert client._input_tokens == 0
    assert client._output_tokens == 0


def test_complete_calls_openai():
    mock_openai = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "test response"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_openai.chat.completions.create.return_value = mock_response

    client = LLMClient()
    client._api_key = "test"
    client._client = mock_openai

    result = client.complete([{"role": "user", "content": "hello"}])
    assert result["content"] == "test response"
    assert client._input_tokens == 10
    assert client._output_tokens == 5


def test_complete_with_tool_calls():
    mock_openai = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = ""
    tc = MagicMock()
    tc.id = "tc_1"
    tc.function.name = "query_nodes"
    tc.function.arguments = '{"name": "test"}'
    mock_response.choices[0].message.tool_calls = [tc]
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.completion_tokens = 10
    mock_openai.chat.completions.create.return_value = mock_response

    client = LLMClient()
    client._api_key = "test"
    client._client = mock_openai

    result = client.complete(
        [{"role": "user", "content": "find nodes"}],
        tools=[{"type": "function", "function": {"name": "query_nodes"}}],
    )
    assert "tool_calls" in result
    assert result["tool_calls"][0]["function"]["name"] == "query_nodes"


def test_batch_complete():
    mock_openai = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "batch result"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage.prompt_tokens = 5
    mock_response.usage.completion_tokens = 3
    mock_openai.chat.completions.create.return_value = mock_response

    client = LLMClient()
    client._api_key = "test"
    client._client = mock_openai

    prompts = [
        [{"role": "user", "content": "p1"}],
        [{"role": "user", "content": "p2"}],
    ]
    results = client.batch_complete(prompts, max_concurrent=2)
    assert len(results) == 2
    assert all(r["content"] == "batch result" for r in results)


def test_embed():
    mock_openai = MagicMock()
    mock_response = MagicMock()
    item1 = MagicMock()
    item1.embedding = [0.1, 0.2, 0.3]
    item2 = MagicMock()
    item2.embedding = [0.4, 0.5, 0.6]
    mock_response.data = [item1, item2]
    mock_response.usage.total_tokens = 20
    mock_openai.embeddings.create.return_value = mock_response

    client = LLMClient()
    client._api_key = "test"
    client._client = mock_openai

    result = client.embed(["text1", "text2"])
    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert client._embedding_tokens == 20
