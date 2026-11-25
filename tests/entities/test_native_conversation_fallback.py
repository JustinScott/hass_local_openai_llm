"""Tests for native conversation fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from custom_components.local_openai.const import CONF_NATIVE_CONVERSATION_FALLBACK
from custom_components.local_openai.conversation import LocalAiConversationEntity
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent


@pytest.mark.parametrize(
    ("enabled", "error_code", "executed", "fallback"),
    [
        (True, None, False, False),
        (True, intent.IntentResponseErrorCode.NO_INTENT_MATCH, False, True),
        (True, intent.IntentResponseErrorCode.NO_VALID_TARGETS, False, True),
        (True, intent.IntentResponseErrorCode.NO_VALID_TARGETS, True, False),
        (True, intent.IntentResponseErrorCode.FAILED_TO_HANDLE, False, False),
        (True, intent.IntentResponseErrorCode.UNKNOWN, False, False),
        (False, None, False, True),
        (None, None, False, True),
    ],
)
async def test_native_conversation_fallback(
    hass: HomeAssistant,
    enabled: bool | None,
    error_code: intent.IntentResponseErrorCode | None,
    executed: bool,
    fallback: bool,
) -> None:
    """Test native handling and safe fallback to the LLM."""
    entry = MagicMock()
    entry.data = {}
    entry.runtime_data = MagicMock()
    subentry = MagicMock()
    subentry.data = {"model": "test-model"}
    if enabled is not None:
        subentry.data[CONF_NATIVE_CONVERSATION_FALLBACK] = enabled
    entity = LocalAiConversationEntity(entry, subentry)
    entity.hass = hass
    entity._async_handle_chat_log = AsyncMock(return_value=False)

    user_input = MagicMock(spec=conversation.ConversationInput)
    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.content = []
    chat_log.conversation_id = "test_conversation_id"
    chat_log.async_provide_llm_data = AsyncMock()
    response = intent.IntentResponse(language="en")
    if error_code is not None:
        response.async_set_error(error_code, "Error")
    native_result = conversation.ConversationResult(
        response=response, conversation_id=chat_log.conversation_id
    )
    llm_result = MagicMock(spec=conversation.ConversationResult)

    async def native_converse(
        *_args: object, **_kwargs: object
    ) -> conversation.ConversationResult:
        if executed:
            chat_log.content.append(MagicMock(spec=conversation.ToolResultContent))
        chat_log.content.append(
            conversation.AssistantContent(agent_id="homeassistant", content="Native")
        )
        return native_result

    with (
        patch(
            "custom_components.local_openai.conversation.conversation.async_converse",
            side_effect=native_converse,
        ) as converse,
        patch(
            "custom_components.local_openai.conversation.llm.async_get_apis",
            return_value=[],
        ),
        patch(
            "custom_components.local_openai.conversation.conversation.async_get_result_from_chat_log",
            return_value=llm_result,
        ),
    ):
        result = await entity._async_handle_message(user_input, chat_log)

    assert converse.await_count == int(bool(enabled))
    assert entity._async_handle_chat_log.await_count == int(fallback)
    assert result is (llm_result if fallback else native_result)
