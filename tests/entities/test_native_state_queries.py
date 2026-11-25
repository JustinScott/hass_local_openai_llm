"""Tests for the optional native state-query fast path."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from custom_components.local_openai.const import CONF_EXTENDED_LOCAL_STATE_QUERIES
from custom_components.local_openai.conversation import LocalAiConversationEntity
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent


@pytest.fixture
def local_entity(hass: HomeAssistant) -> LocalAiConversationEntity:
    """Create a conversation entity without a live LLM server."""
    entry = MagicMock()
    entry.data = {}
    subentry = MagicMock()
    subentry.data = {"model": "test-model", CONF_EXTENDED_LOCAL_STATE_QUERIES: True}
    entity = LocalAiConversationEntity(entry, subentry)
    entity.hass = hass
    return entity


@pytest.fixture
def input_and_log() -> tuple[conversation.ConversationInput, conversation.ChatLog]:
    """Make the objects handed to the conversation entity."""
    user_input = MagicMock(spec=conversation.ConversationInput)
    user_input.agent_id = "conversation.test_local"
    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "conversation-123"
    return user_input, chat_log


async def test_native_state_query_bypasses_llm(
    local_entity: LocalAiConversationEntity,
    input_and_log: tuple[conversation.ConversationInput, conversation.ChatLog],
) -> None:
    """Only HassGetState is executed, and its speech becomes final."""
    user_input, chat_log = input_and_log
    native_response = intent.IntentResponse(language="en")
    native_response.async_set_speech("No, the bedroom light is off.")
    native_response.response_type = intent.IntentResponseType.QUERY_ANSWER

    async def handle_intents(
        received_input: conversation.ConversationInput,
        received_log: conversation.ChatLog,
        *,
        intent_filter: Callable[[SimpleNamespace], bool],
    ) -> intent.IntentResponse:
        assert received_input is user_input
        assert received_log is chat_log
        assert not intent_filter(
            SimpleNamespace(intent=SimpleNamespace(name=intent.INTENT_GET_STATE))
        )
        assert intent_filter(
            SimpleNamespace(intent=SimpleNamespace(name="HassTurnOff"))
        )
        return native_response

    local_entity._async_handle_chat_log = AsyncMock(return_value=False)
    chat_log.async_provide_llm_data = AsyncMock()
    native_agent = MagicMock()
    native_agent.async_handle_intents = AsyncMock(side_effect=handle_intents)
    with patch(
        "custom_components.local_openai.conversation.conversation.async_get_agent",
        return_value=native_agent,
    ):
        result = await local_entity._async_handle_message(user_input, chat_log)

    local_entity._async_handle_chat_log.assert_not_awaited()
    chat_log.async_provide_llm_data.assert_not_awaited()
    assert result is not None
    assert result.response is native_response
    assert result.conversation_id == "conversation-123"
    assert result.continue_conversation is False
    chat_log.async_add_assistant_content_without_tools.assert_called_once()
    assistant = chat_log.async_add_assistant_content_without_tools.call_args.args[0]
    assert assistant.content == "No, the bedroom light is off."


@pytest.mark.parametrize("outcome", ["unmatched", "error", "exception", "missing_api"])
async def test_native_query_falls_back(
    local_entity: LocalAiConversationEntity,
    input_and_log: tuple[conversation.ConversationInput, conversation.ChatLog],
    outcome: str,
) -> None:
    """Unmatched or failed local queries leave the LLM fallback available."""
    user_input, chat_log = input_and_log
    agent = MagicMock()
    if outcome == "unmatched":
        agent.async_handle_intents = AsyncMock(return_value=None)
    elif outcome == "error":
        response = intent.IntentResponse(language="en")
        response.response_type = intent.IntentResponseType.ERROR
        agent.async_handle_intents = AsyncMock(return_value=response)
    elif outcome == "exception":
        agent.async_handle_intents = AsyncMock(side_effect=RuntimeError("offline"))
    else:
        agent = object()

    with patch(
        "custom_components.local_openai.conversation.conversation.async_get_agent",
        return_value=agent,
    ):
        result = await local_entity._async_try_native_state_query(
            user_input, chat_log, False
        )

    assert result is None
    chat_log.async_add_assistant_content_without_tools.assert_not_called()


async def test_fast_path_disabled_by_default(
    local_entity: LocalAiConversationEntity,
    input_and_log: tuple[conversation.ConversationInput, conversation.ChatLog],
) -> None:
    """Without explicit opt-in, processing continues to the LLM."""
    local_entity.subentry.data = {"model": "test-model"}
    user_input, chat_log = input_and_log
    chat_log.async_provide_llm_data = AsyncMock()
    chat_log.llm_api = MagicMock()
    local_entity._async_handle_chat_log = AsyncMock(return_value=False)

    with (
        patch.object(
            local_entity, "_async_try_native_state_query", new_callable=AsyncMock
        ) as native,
        patch(
            "custom_components.local_openai.conversation.llm.async_get_apis",
            return_value=[],
        ),
        patch(
            "custom_components.local_openai.conversation.conversation.async_get_result_from_chat_log"
        ) as get_result,
    ):
        await local_entity._async_handle_message(user_input, chat_log)

    native.assert_not_awaited()
    local_entity._async_handle_chat_log.assert_awaited_once()
    get_result.assert_called_once()
