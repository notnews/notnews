"""Exercise Claude request serialization through the installed SDK."""

import json

import httpx2
import pandas as pd
import pytest
from anthropic import Anthropic

from notnews import llm


@pytest.mark.parametrize("model", [None, "claude-sonnet-4-6"])
def test_classification_uses_supported_sdk_request(monkeypatch, model):
    requests = []
    expected_model = model or "claude-haiku-4-5-20251001"

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == "/v1/messages"
        assert payload["model"] == expected_model
        assert "temperature" not in payload
        assert payload["messages"][0]["role"] == "user"
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": expected_model,
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "category": "hard_news",
                                "confidence": 0.9,
                                "reasoning": "Economic policy coverage",
                            }
                        ),
                    }
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
        )

    with Anthropic(
        api_key="test-key",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as client:
        monkeypatch.setattr(llm, "Anthropic", lambda **_kwargs: client)
        result = llm.classify_with_llm(
            pd.DataFrame({"text": ["The government announced an economic policy."]}),
            provider="claude",
            api_key="test-key",
            model=model,
        )

    assert len(requests) == 1
    assert result.loc[0, "llm_category"] == "hard_news"
    assert result.loc[0, "llm_confidence"] == 0.9
    assert result.loc[0, "llm_reasoning"] == "Economic policy coverage"
