from __future__ import annotations

from typing import Any, AsyncGenerator, Mapping, Sequence, Optional, Union, List

from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelFamily,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    FunctionExecutionResultMessage,
)
from autogen_core.tools import Tool, ToolSchema
from autogen_core import CancellationToken, Component
from pydantic import BaseModel
from typing_extensions import Self

from ...harness.tsce_chat import TSCEChat


class TSCEChatCompletionClientConfig(BaseModel):
    model: Optional[str] = None


class TSCEChatCompletionClient(ChatCompletionClient, Component[TSCEChatCompletionClientConfig]):
    """Chat completion client that routes requests through :class:`TSCEChat`."""

    __protocol__ = ChatCompletionClient
    component_type = "tsce_chat_completion_client"
    component_provider_override = "autogen_ext.models.tsce.TSCEChatCompletionClient"
    component_config_schema = TSCEChatCompletionClientConfig

    def __init__(self, model: str | None = None) -> None:
        self._tsce = TSCEChat(model=model)
        self._model_info = ModelInfo(
            vision=False,
            function_calling=False,
            json_output=False,
            family=ModelFamily.UNKNOWN,
            structured_output=False,
        )
        self._cur_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    # ------------------------------------------------------------------
    def _convert_messages(self, messages: Sequence[LLMMessage]) -> List[dict[str, str]]:
        chat: List[dict[str, str]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                chat.append({"role": "system", "content": m.content})
            elif isinstance(m, UserMessage):
                content = m.content if isinstance(m.content, str) else ""
                chat.append({"role": "user", "content": content})
            elif isinstance(m, AssistantMessage):
                if isinstance(m.content, str):
                    chat.append({"role": "assistant", "content": m.content})
            elif isinstance(m, FunctionExecutionResultMessage):
                # ignore tool results for now
                pass
        return chat

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        chat = self._convert_messages(messages)
        reply = self._tsce(chat)
        content = reply.content
        tokens = content.split()
        self._cur_usage = RequestUsage(prompt_tokens=0, completion_tokens=len(tokens))
        self._total_usage.prompt_tokens += self._cur_usage.prompt_tokens
        self._total_usage.completion_tokens += self._cur_usage.completion_tokens
        return CreateResult(
            finish_reason="stop",
            content=content,
            usage=self._cur_usage,
            cached=False,
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        result = await self.create(
            messages,
            tools=tools,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        for i, token in enumerate(result.content.split()):
            if i < len(result.content.split()) - 1:
                yield token + " "
            else:
                yield token
        yield result

    async def close(self) -> None:
        pass

    def actual_usage(self) -> RequestUsage:
        return self._cur_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 0

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 0

    @property
    def capabilities(self) -> ModelInfo:  # type: ignore
        return self._model_info

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    def _to_config(self) -> TSCEChatCompletionClientConfig:
        return TSCEChatCompletionClientConfig(model=self._tsce.model)

    @classmethod
    def _from_config(cls, config: TSCEChatCompletionClientConfig) -> Self:
        return cls(model=config.model)
