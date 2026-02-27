#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for context summarization feature."""

import asyncio
import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import LLMContextSummaryRequestFrame, LLMContextSummaryResultFrame
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.services.llm_service import LLMService
from pipecat.utils.context.llm_context_summarization import (
    LLMContextSummarizationConfig,
    LLMContextSummarizationUtil,
)


class TestContextSummarizationMixin(unittest.TestCase):
    """Tests for LLMContextSummarizationUtil."""

    def test_estimate_tokens_simple_text(self):
        """Test token estimation with simple text."""
        # Simple sentence: "Hello world" = 11 chars / 4 = 2.75 -> 2 tokens
        tokens = LLMContextSummarizationUtil.estimate_tokens("Hello world")
        self.assertEqual(tokens, 2)

        # More words: "This is a test message" = 22 chars / 4 = 5.5 -> 5 tokens
        tokens = LLMContextSummarizationUtil.estimate_tokens("This is a test message")
        self.assertEqual(tokens, 5)

    def test_estimate_tokens_empty(self):
        """Test token estimation with empty text."""
        tokens = LLMContextSummarizationUtil.estimate_tokens("")
        self.assertEqual(tokens, 0)

    def test_estimate_context_tokens(self):
        """Test context token estimation."""
        context = LLMContext()

        # Empty context
        self.assertEqual(LLMContextSummarizationUtil.estimate_context_tokens(context), 0)

        # Add messages
        context.add_message({"role": "system", "content": "You are helpful"})  # ~4 words
        context.add_message({"role": "user", "content": "Hello"})  # ~1 word
        context.add_message({"role": "assistant", "content": "Hi there"})  # ~2 words

        # Each message has ~10 token overhead
        # Total content: ~7 words * 1.3 = ~9 tokens
        # Total overhead: 3 * 10 = 30 tokens
        # Expected: ~39 tokens
        total = LLMContextSummarizationUtil.estimate_context_tokens(context)
        self.assertGreater(total, 30)  # At least overhead
        self.assertLess(total, 50)  # Not too much

    def test_get_messages_to_summarize_basic(self):
        """Test basic message extraction for summarization."""
        context = LLMContext()

        # Add messages
        context.add_message({"role": "system", "content": "System prompt"})
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message({"role": "user", "content": "Message 2"})
        context.add_message({"role": "assistant", "content": "Response 2"})
        context.add_message({"role": "user", "content": "Message 3"})
        context.add_message({"role": "assistant", "content": "Response 3"})

        # Keep last 2 messages
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 2)

        # Get first system message from context
        first_system = None
        for msg in context.messages:
            if msg.get("role") == "system":
                first_system = msg
                break

        # Should get system message
        self.assertIsNotNone(first_system)
        self.assertEqual(first_system["content"], "System prompt")

        # Should get middle messages (indices 1-4)
        self.assertEqual(len(result.messages), 4)
        self.assertEqual(result.messages[0]["content"], "Message 1")
        self.assertEqual(result.messages[-1]["content"], "Response 2")

        # Last index should be 4 (0-indexed)
        self.assertEqual(result.last_summarized_index, 4)

    def test_get_messages_to_summarize_no_system(self):
        """Test message extraction when there's no system message."""
        context = LLMContext()

        # Add messages without system prompt
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message({"role": "user", "content": "Message 2"})
        context.add_message({"role": "assistant", "content": "Response 2"})

        # Keep last 1 message
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 1)

        # Get first system message from context
        first_system = None
        for msg in context.messages:
            if msg.get("role") == "system":
                first_system = msg
                break

        # Should have no system message
        self.assertIsNone(first_system)

        # Should get first 3 messages
        self.assertEqual(len(result.messages), 3)
        self.assertEqual(result.last_summarized_index, 2)

    def test_get_messages_to_summarize_insufficient(self):
        """Test when there aren't enough messages to summarize."""
        context = LLMContext()

        # Add only 2 messages
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})

        # Try to keep 2 messages (same as total)
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 2)

        # Should return empty
        self.assertEqual(len(result.messages), 0)
        self.assertEqual(result.last_summarized_index, -1)

    def test_format_messages_for_summary(self):
        """Test message formatting for summary."""

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

        transcript = LLMContextSummarizationUtil.format_messages_for_summary(messages)

        self.assertIn("USER: Hello", transcript)
        self.assertIn("ASSISTANT: Hi there", transcript)
        self.assertIn("USER: How are you?", transcript)

    def test_format_messages_with_list_content(self):
        """Test formatting messages with list content."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"},
                ],
            }
        ]

        transcript = LLMContextSummarizationUtil.format_messages_for_summary(messages)

        self.assertIn("USER: First part Second part", transcript)


class TestLLMContextSummarizationConfig(unittest.TestCase):
    """Tests for LLMContextSummarizationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMContextSummarizationConfig()

        self.assertEqual(config.max_context_tokens, 8000)
        self.assertEqual(config.max_unsummarized_messages, 20)
        self.assertEqual(config.min_messages_after_summary, 4)
        self.assertIsNone(config.summarization_prompt)

    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMContextSummarizationConfig(
            max_context_tokens=2500,
            target_context_tokens=2000,
            max_unsummarized_messages=15,
            min_messages_after_summary=4,
            summarization_prompt="Custom prompt",
        )

        self.assertEqual(config.max_context_tokens, 2500)
        self.assertEqual(config.target_context_tokens, 2000)
        self.assertEqual(config.max_unsummarized_messages, 15)
        self.assertEqual(config.min_messages_after_summary, 4)
        self.assertEqual(config.summary_prompt, "Custom prompt")

    def test_summary_prompt_property(self):
        """Test summary_prompt property uses default when None."""
        config = LLMContextSummarizationConfig()
        self.assertIn("summarizing a conversation", config.summary_prompt.lower())

        config_with_custom = LLMContextSummarizationConfig(summarization_prompt="Custom")
        self.assertEqual(config_with_custom.summary_prompt, "Custom")


class TestFunctionCallHandling(unittest.TestCase):
    """Tests for function call handling in summarization."""

    def test_function_call_in_progress_not_summarized(self):
        """Test that messages with function calls in progress are not summarized."""
        context = LLMContext()

        # Add messages including a function call without result
        context.add_message({"role": "system", "content": "System prompt"})
        context.add_message({"role": "user", "content": "What time is it?"})
        context.add_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ],
            }
        )
        # No tool result yet - function call is in progress
        context.add_message({"role": "user", "content": "Latest message"})

        # Try to keep last 1 message
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 1)

        # Should only get the first user message, stopping before the function call
        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0]["content"], "What time is it?")
        self.assertEqual(result.last_summarized_index, 1)

    def test_completed_function_call_can_be_summarized(self):
        """Test that completed function calls can be summarized."""
        context = LLMContext()

        # Add messages including a complete function call sequence
        context.add_message({"role": "system", "content": "System prompt"})
        context.add_message({"role": "user", "content": "What time is it?"})
        context.add_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ],
            }
        )
        # Tool result completes the function call
        context.add_message(
            {"role": "tool", "tool_call_id": "call_123", "content": '{"time": "10:30 AM"}'}
        )
        context.add_message({"role": "assistant", "content": "It's 10:30 AM"})
        context.add_message({"role": "user", "content": "Latest message"})

        # Try to keep last 1 message
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 1)

        # Should get all messages except the last one (complete function call is included)
        self.assertEqual(len(result.messages), 4)
        self.assertEqual(result.messages[0]["content"], "What time is it?")
        self.assertEqual(result.messages[-1]["content"], "It's 10:30 AM")
        self.assertEqual(result.last_summarized_index, 4)

    def test_multiple_function_calls_in_progress(self):
        """Test handling of multiple function calls in progress."""
        context = LLMContext()

        # Add messages with multiple function calls
        context.add_message({"role": "system", "content": "System prompt"})
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message({"role": "user", "content": "What's the time and date?"})
        context.add_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_time",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    },
                    {
                        "id": "call_date",
                        "type": "function",
                        "function": {"name": "get_date", "arguments": "{}"},
                    },
                ],
            }
        )
        # Only one tool result - other call still in progress
        context.add_message(
            {"role": "tool", "tool_call_id": "call_time", "content": '{"time": "10:30 AM"}'}
        )
        context.add_message({"role": "user", "content": "Latest message"})

        # Try to keep last 1 message
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 1)

        # Should stop before the function call that's in progress
        # Messages to summarize: indices 1, 2, 3 (stops before index 4 where incomplete call is)
        self.assertEqual(len(result.messages), 3)
        self.assertEqual(result.last_summarized_index, 3)

    def test_multiple_completed_function_calls(self):
        """Test that multiple completed function calls can be summarized."""
        context = LLMContext()

        # Add messages with multiple completed function calls
        context.add_message({"role": "system", "content": "System prompt"})
        context.add_message({"role": "user", "content": "What's the time and date?"})
        context.add_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_time",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    },
                    {
                        "id": "call_date",
                        "type": "function",
                        "function": {"name": "get_date", "arguments": "{}"},
                    },
                ],
            }
        )
        # Both tool results provided
        context.add_message(
            {"role": "tool", "tool_call_id": "call_time", "content": '{"time": "10:30 AM"}'}
        )
        context.add_message(
            {
                "role": "tool",
                "tool_call_id": "call_date",
                "content": '{"date": "January 1, 2024"}',
            }
        )
        context.add_message({"role": "assistant", "content": "It's 10:30 AM on January 1, 2024"})
        context.add_message({"role": "user", "content": "Latest message"})

        # Try to keep last 1 message
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 1)

        # Should get all messages except the last one (all function calls completed)
        self.assertEqual(len(result.messages), 5)
        self.assertEqual(result.last_summarized_index, 5)

    def test_sequential_function_calls_mixed_completion(self):
        """Test sequential function calls with mixed completion states."""
        context = LLMContext()

        # Add messages with sequential function calls
        context.add_message({"role": "system", "content": "System prompt"})

        # First function call - completed
        context.add_message({"role": "user", "content": "What time is it?"})
        context.add_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ],
            }
        )
        context.add_message(
            {"role": "tool", "tool_call_id": "call_1", "content": '{"time": "10:30 AM"}'}
        )
        context.add_message({"role": "assistant", "content": "It's 10:30 AM"})

        # Second function call - in progress
        context.add_message({"role": "user", "content": "What's the date?"})
        context.add_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_date", "arguments": "{}"},
                    }
                ],
            }
        )
        # No result for call_2 yet
        context.add_message({"role": "user", "content": "Latest message"})

        # Try to keep last 1 message
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 1)

        # Should get messages up to and including the first completed function call
        # but stop before the second function call that's in progress
        # Messages to summarize: indices 1, 2, 3, 4, 5 (stops before index 6 where incomplete call is)
        self.assertEqual(len(result.messages), 5)
        self.assertEqual(result.messages[-1]["content"], "What's the date?")
        self.assertEqual(result.last_summarized_index, 5)

    def test_function_call_formatting_in_transcript(self):
        """Test that function calls are properly formatted in transcript."""

        messages = [
            {"role": "user", "content": "What time is it?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": '{"time": "10:30 AM"}'},
            {"role": "assistant", "content": "It's 10:30 AM"},
        ]

        transcript = LLMContextSummarizationUtil.format_messages_for_summary(messages)

        # Check that function call is included
        self.assertIn("TOOL_CALL: get_time({})", transcript)
        # Check that tool result is included
        self.assertIn('TOOL_RESULT[call_123]: {"time": "10:30 AM"}', transcript)

    def test_no_function_calls(self):
        """Test that summarization works normally without function calls."""
        context = LLMContext()

        # Add normal conversation without function calls
        context.add_message({"role": "system", "content": "System prompt"})
        context.add_message({"role": "user", "content": "Hello"})
        context.add_message({"role": "assistant", "content": "Hi"})
        context.add_message({"role": "user", "content": "How are you?"})
        context.add_message({"role": "assistant", "content": "I'm good"})
        context.add_message({"role": "user", "content": "Latest message"})

        # Try to keep last 1 message
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 1)

        # Should get all messages except the last one
        self.assertEqual(len(result.messages), 4)
        self.assertEqual(result.last_summarized_index, 4)


class TestSummaryGenerationExceptions(unittest.IsolatedAsyncioTestCase):
    """Tests for summary generation exception handling."""

    async def test_generate_summary_raises_on_no_messages(self):
        """Test that _generate_summary raises RuntimeError when there are no messages to summarize."""
        llm_service = LLMService()
        context = LLMContext()

        # Add only one message (system), which isn't enough to summarize
        context.add_message({"role": "system", "content": "System prompt"})

        frame = LLMContextSummaryRequestFrame(
            request_id="test",
            context=context,
            min_messages_to_keep=1,
            target_context_tokens=1000,
            summarization_prompt="Summarize this",
        )

        with self.assertRaises(RuntimeError) as cm:
            await llm_service._generate_summary(frame)

        self.assertEqual(str(cm.exception), "No messages to summarize")

    async def test_generate_summary_raises_on_no_run_inference(self):
        """Test that _generate_summary raises RuntimeError when run_inference is not implemented."""
        # Create a minimal LLM service - base class raises NotImplementedError
        llm_service = LLMService()

        context = LLMContext()
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message({"role": "user", "content": "Message 2"})

        frame = LLMContextSummaryRequestFrame(
            request_id="test",
            context=context,
            min_messages_to_keep=1,
            target_context_tokens=1000,
            summarization_prompt="Summarize this",
        )

        with self.assertRaises(RuntimeError) as cm:
            await llm_service._generate_summary(frame)

        self.assertIn("does not implement run_inference", str(cm.exception))
        self.assertIn("LLMService", str(cm.exception))

    async def test_generate_summary_raises_on_empty_response(self):
        """Test that _generate_summary raises RuntimeError when LLM returns empty summary."""
        llm_service = LLMService()
        # Mock run_inference to return None
        llm_service.run_inference = AsyncMock(return_value=None)

        context = LLMContext()
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message({"role": "user", "content": "Message 2"})

        frame = LLMContextSummaryRequestFrame(
            request_id="test",
            context=context,
            min_messages_to_keep=1,
            target_context_tokens=1000,
            summarization_prompt="Summarize this",
        )

        with self.assertRaises(RuntimeError) as cm:
            await llm_service._generate_summary(frame)

        self.assertEqual(str(cm.exception), "LLM returned empty summary")

    async def test_generate_summary_task_handles_exceptions(self):
        """Test that _generate_summary_task properly handles exceptions from _generate_summary."""
        llm_service = LLMService()

        # Mock broadcast_frame to capture the result
        broadcast_calls = []

        async def mock_broadcast(frame_class, **kwargs):
            broadcast_calls.append((frame_class, kwargs))

        llm_service.broadcast_frame = mock_broadcast

        # Mock push_error
        llm_service.push_error = AsyncMock()

        context = LLMContext()
        context.add_message({"role": "system", "content": "System prompt"})

        frame = LLMContextSummaryRequestFrame(
            request_id="test_123",
            context=context,
            min_messages_to_keep=1,
            target_context_tokens=1000,
            summarization_prompt="Summarize this",
        )

        # Execute the task
        await llm_service._generate_summary_task(frame)

        # Verify broadcast_frame was called with error
        self.assertEqual(len(broadcast_calls), 1)
        frame_class, kwargs = broadcast_calls[0]
        self.assertEqual(kwargs["request_id"], "test_123")
        self.assertEqual(kwargs["summary"], "")
        self.assertEqual(kwargs["last_summarized_index"], -1)
        self.assertEqual(
            kwargs["error"], "Error generating context summary: No messages to summarize"
        )

        # Verify push_error was called
        llm_service.push_error.assert_called_once()

    async def test_generate_summary_success(self):
        """Test that _generate_summary returns successfully with valid input."""
        llm_service = LLMService()
        # Mock run_inference to return a summary
        llm_service.run_inference = AsyncMock(return_value="This is a summary of the conversation")

        context = LLMContext()
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message({"role": "user", "content": "Message 2"})

        frame = LLMContextSummaryRequestFrame(
            request_id="test",
            context=context,
            min_messages_to_keep=1,
            target_context_tokens=1000,
            summarization_prompt="Summarize this",
        )

        summary, last_index = await llm_service._generate_summary(frame)

        self.assertEqual(summary, "This is a summary of the conversation")
        self.assertGreater(last_index, -1)
        self.assertEqual(last_index, 1)  # Should be the index of the last summarized message

    async def test_generate_summary_task_timeout(self):
        """Test that _generate_summary_task handles timeout correctly."""
        llm_service = LLMService()

        # Mock _generate_summary to hang
        async def slow_summary(frame):
            await asyncio.sleep(10)
            return ("summary", 1)

        llm_service._generate_summary = slow_summary

        broadcast_calls = []

        async def mock_broadcast(frame_class, **kwargs):
            broadcast_calls.append((frame_class, kwargs))

        llm_service.broadcast_frame = mock_broadcast
        llm_service.push_error = AsyncMock()

        context = LLMContext()
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message({"role": "user", "content": "Message 2"})

        frame = LLMContextSummaryRequestFrame(
            request_id="timeout_test",
            context=context,
            min_messages_to_keep=1,
            target_context_tokens=1000,
            summarization_prompt="Summarize this",
            summarization_timeout=0.1,  # Very short timeout
        )

        await llm_service._generate_summary_task(frame)

        # Should have broadcast an error result
        self.assertEqual(len(broadcast_calls), 1)
        _, kwargs = broadcast_calls[0]
        self.assertEqual(kwargs["request_id"], "timeout_test")
        self.assertEqual(kwargs["summary"], "")
        self.assertEqual(kwargs["last_summarized_index"], -1)
        # error is None for timeout path (push_error is called instead)
        self.assertIsNone(kwargs["error"])

        # push_error should have been called with timeout message
        llm_service.push_error.assert_called_once()
        call_args = llm_service.push_error.call_args
        error_msg = call_args.kwargs.get("error_msg") or call_args.args[0]
        self.assertIn("timed out", error_msg)


class TestDedicatedLLMSummarization(unittest.IsolatedAsyncioTestCase):
    """Tests for dedicated LLM summarization in LLMAssistantAggregator."""

    def _create_context_and_frame(self):
        """Create a context with enough messages and a matching request frame."""
        context = LLMContext()
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message({"role": "user", "content": "Message 2"})

        frame = LLMContextSummaryRequestFrame(
            request_id="dedicated_test",
            context=context,
            min_messages_to_keep=1,
            target_context_tokens=1000,
            summarization_prompt="Summarize this",
            summarization_timeout=5.0,
        )
        return context, frame

    async def test_dedicated_llm_success(self):
        """Test that dedicated LLM generates summary and feeds result to summarizer."""
        from pipecat.processors.aggregators.llm_context_summarizer import LLMContextSummarizer
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMAssistantAggregator,
            LLMAssistantAggregatorParams,
        )
        from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

        context, frame = self._create_context_and_frame()

        # Create a mock dedicated LLM
        dedicated_llm = LLMService()
        dedicated_llm._generate_summary = AsyncMock(return_value=("Dedicated summary", 1))

        config = LLMContextSummarizationConfig(
            max_context_tokens=50,
            llm=dedicated_llm,
        )
        params = LLMAssistantAggregatorParams(
            enable_context_summarization=True,
            context_summarization_config=config,
        )
        aggregator = LLMAssistantAggregator(context, params=params)

        # Mock summarizer.process_frame to capture the result
        result_frames = []
        original_process = aggregator._summarizer.process_frame

        async def capture_process(frame):
            result_frames.append(frame)
            await original_process(frame)

        aggregator._summarizer.process_frame = capture_process

        # Call the method directly
        await aggregator._generate_summary_with_dedicated_llm(dedicated_llm, frame)

        # Verify the dedicated LLM was called
        dedicated_llm._generate_summary.assert_called_once_with(frame)

        # Verify result was fed to the summarizer
        self.assertEqual(len(result_frames), 1)
        result = result_frames[0]
        self.assertIsInstance(result, LLMContextSummaryResultFrame)
        self.assertEqual(result.request_id, "dedicated_test")
        self.assertEqual(result.summary, "Dedicated summary")
        self.assertEqual(result.last_summarized_index, 1)
        self.assertIsNone(result.error)

    async def test_dedicated_llm_timeout(self):
        """Test that dedicated LLM timeout produces error result."""
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMAssistantAggregator,
            LLMAssistantAggregatorParams,
        )

        context, _ = self._create_context_and_frame()

        # Create a mock dedicated LLM that hangs
        dedicated_llm = LLMService()

        async def slow_summary(frame):
            await asyncio.sleep(10)
            return ("summary", 1)

        dedicated_llm._generate_summary = slow_summary

        config = LLMContextSummarizationConfig(
            max_context_tokens=50,
            llm=dedicated_llm,
        )
        params = LLMAssistantAggregatorParams(
            enable_context_summarization=True,
            context_summarization_config=config,
        )
        aggregator = LLMAssistantAggregator(context, params=params)

        # Mock summarizer.process_frame to capture the result
        result_frames = []

        async def capture_process(frame):
            result_frames.append(frame)

        aggregator._summarizer.process_frame = capture_process

        # Create frame with very short timeout
        frame = LLMContextSummaryRequestFrame(
            request_id="timeout_test",
            context=context,
            min_messages_to_keep=1,
            target_context_tokens=1000,
            summarization_prompt="Summarize this",
            summarization_timeout=0.1,
        )

        await aggregator._generate_summary_with_dedicated_llm(dedicated_llm, frame)

        # Verify error result was fed to summarizer
        self.assertEqual(len(result_frames), 1)
        result = result_frames[0]
        self.assertIsInstance(result, LLMContextSummaryResultFrame)
        self.assertEqual(result.request_id, "timeout_test")
        self.assertEqual(result.summary, "")
        self.assertEqual(result.last_summarized_index, -1)
        self.assertIn("timed out", result.error)

    async def test_dedicated_llm_exception(self):
        """Test that dedicated LLM exceptions produce error result."""
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMAssistantAggregator,
            LLMAssistantAggregatorParams,
        )

        context, frame = self._create_context_and_frame()

        # Create a mock dedicated LLM that raises
        dedicated_llm = LLMService()
        dedicated_llm._generate_summary = AsyncMock(
            side_effect=RuntimeError("LLM connection failed")
        )

        config = LLMContextSummarizationConfig(
            max_context_tokens=50,
            llm=dedicated_llm,
        )
        params = LLMAssistantAggregatorParams(
            enable_context_summarization=True,
            context_summarization_config=config,
        )
        aggregator = LLMAssistantAggregator(context, params=params)
        aggregator.push_error = AsyncMock()

        # Mock summarizer.process_frame to capture the result
        result_frames = []

        async def capture_process(frame):
            result_frames.append(frame)

        aggregator._summarizer.process_frame = capture_process

        await aggregator._generate_summary_with_dedicated_llm(dedicated_llm, frame)

        # Verify error result was fed to summarizer
        self.assertEqual(len(result_frames), 1)
        result = result_frames[0]
        self.assertIsInstance(result, LLMContextSummaryResultFrame)
        self.assertEqual(result.request_id, "dedicated_test")
        self.assertEqual(result.summary, "")
        self.assertEqual(result.last_summarized_index, -1)
        self.assertIn("LLM connection failed", result.error)

        # push_error should have been called
        aggregator.push_error.assert_called_once()

    async def test_on_request_summarization_routes_to_dedicated_llm(self):
        """Test that _on_request_summarization routes to dedicated LLM when configured."""
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMAssistantAggregator,
            LLMAssistantAggregatorParams,
        )

        context, frame = self._create_context_and_frame()

        dedicated_llm = LLMService()
        dedicated_llm._generate_summary = AsyncMock(return_value=("Summary", 1))

        config = LLMContextSummarizationConfig(
            max_context_tokens=50,
            llm=dedicated_llm,
        )
        params = LLMAssistantAggregatorParams(
            enable_context_summarization=True,
            context_summarization_config=config,
        )
        aggregator = LLMAssistantAggregator(context, params=params)
        aggregator.push_frame = AsyncMock()

        # Track what coroutine is passed to create_task
        created_coros = []
        original_create_task = aggregator.create_task

        def mock_create_task(coro, *args, **kwargs):
            created_coros.append(coro)
            # Actually run the coroutine to avoid "never awaited" warning
            task = asyncio.ensure_future(coro)
            return task

        aggregator.create_task = mock_create_task

        await aggregator._on_request_summarization(aggregator._summarizer, frame)

        # Should NOT push frame upstream
        aggregator.push_frame.assert_not_called()

        # Should have created a task for the dedicated LLM
        self.assertEqual(len(created_coros), 1)

        # Wait for the task to complete
        await asyncio.sleep(0.05)

    async def test_on_request_summarization_pushes_upstream_without_dedicated_llm(self):
        """Test that _on_request_summarization pushes upstream when no dedicated LLM."""
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMAssistantAggregator,
            LLMAssistantAggregatorParams,
        )
        from pipecat.processors.frame_processor import FrameDirection

        context, frame = self._create_context_and_frame()

        config = LLMContextSummarizationConfig(max_context_tokens=50)
        params = LLMAssistantAggregatorParams(
            enable_context_summarization=True,
            context_summarization_config=config,
        )
        aggregator = LLMAssistantAggregator(context, params=params)
        aggregator.push_frame = AsyncMock()

        await aggregator._on_request_summarization(aggregator._summarizer, frame)

        # Should push frame upstream
        aggregator.push_frame.assert_called_once_with(frame, FrameDirection.UPSTREAM)


class TestLLMSpecificMessageHandling(unittest.TestCase):
    """Tests that LLMSpecificMessage objects are correctly skipped in summarization."""

    def test_estimate_context_tokens_skips_specific_messages(self):
        """Test that estimate_context_tokens skips LLMSpecificMessage objects."""
        context = LLMContext()
        context.add_message({"role": "user", "content": "Hello"})
        context.add_message(LLMSpecificMessage(llm="google", message={}))
        context.add_message({"role": "assistant", "content": "Hi there"})

        tokens_with_specific = LLMContextSummarizationUtil.estimate_context_tokens(context)

        context_without = LLMContext()
        context_without.add_message({"role": "user", "content": "Hello"})
        context_without.add_message({"role": "assistant", "content": "Hi there"})
        tokens_without = LLMContextSummarizationUtil.estimate_context_tokens(context_without)

        self.assertEqual(tokens_with_specific, tokens_without)

    def test_get_messages_to_summarize_with_specific_messages(self):
        """Test that get_messages_to_summarize handles LLMSpecificMessage objects."""
        context = LLMContext()
        context.add_message({"role": "system", "content": "System prompt"})
        context.add_message(LLMSpecificMessage(llm="google", message={}))
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message(LLMSpecificMessage(llm="google", message={}))
        context.add_message({"role": "user", "content": "Message 2"})
        context.add_message({"role": "assistant", "content": "Response 2"})

        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 2)

        self.assertEqual(len(result.messages), 4)
        self.assertEqual(result.last_summarized_index, 4)

    def test_format_messages_skips_specific_messages(self):
        """Test that format_messages_for_summary skips LLMSpecificMessage objects."""
        messages = [
            {"role": "user", "content": "Hello"},
            LLMSpecificMessage(llm="google", message={}),
            {"role": "assistant", "content": "Hi there"},
        ]

        transcript = LLMContextSummarizationUtil.format_messages_for_summary(messages)

        self.assertIn("USER: Hello", transcript)
        self.assertIn("ASSISTANT: Hi there", transcript)

    def test_function_call_tracking_skips_specific_messages(self):
        """Test that _get_function_calls_in_progress_index skips LLMSpecificMessage."""
        messages = [
            {"role": "user", "content": "What time is it?"},
            LLMSpecificMessage(llm="google", message={}),
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ],
            },
            LLMSpecificMessage(llm="google", message={}),
            {"role": "tool", "tool_call_id": "call_123", "content": '{"time": "10:30 AM"}'},
        ]

        result = LLMContextSummarizationUtil._get_function_calls_in_progress_index(messages, 0)
        self.assertEqual(result, -1)


if __name__ == "__main__":
    unittest.main()
