#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram speech-to-text service for AWS SageMaker.

This module provides a Pipecat STT service that connects to Deepgram models
deployed on AWS SageMaker endpoints. Uses HTTP/2 bidirectional streaming for
low-latency real-time transcription with support for interim results, multiple
languages, and various Deepgram features.
"""

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Mapping, Optional, Type

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.aws.sagemaker.bidi_client import SageMakerBidiClient
from pipecat.services.settings import _S, NOT_GIVEN, STTSettings, _NotGiven, is_given
from pipecat.services.stt_latency import DEEPGRAM_SAGEMAKER_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from deepgram import LiveOptions
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use DeepgramSageMakerSTTService, you need to `pip install pipecat-ai[deepgram,sagemaker]`."
    )
    raise Exception(f"Missing module: {e}")


@dataclass
class DeepgramSageMakerSTTSettings(STTSettings):
    """Settings for the Deepgram SageMaker STT service.

    Wraps the Deepgram SDK's ``LiveOptions`` in a single ``live_options``
    field.  All Deepgram-specific options (``filler_words``, ``diarize``,
    ``utterance_end_ms``, etc.) should be passed directly via
    ``LiveOptions``.

    In **delta mode** (i.e. when carried by ``STTUpdateSettingsFrame``),
    ``live_options`` is treated as a **delta** — its non-None fields are
    merged into the stored ``LiveOptions``, not replaced wholesale.  For
    example, ``DeepgramSageMakerSTTSettings(live_options=LiveOptions(punctuate=False))``
    changes only ``punctuate`` and leaves all other options intact.

    Parameters:
        live_options: Deepgram ``LiveOptions`` for STT configuration.
            In delta mode only its non-None fields are merged into the
            stored options.
    """

    live_options: LiveOptions | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    # Valid LiveOptions __init__ parameter names (cached at class level).
    _live_options_params: set[str] | None = field(default=None, init=False, repr=False)

    @classmethod
    def _get_live_options_params(cls) -> set[str]:
        """Return the set of valid ``LiveOptions.__init__`` parameter names."""
        if cls._live_options_params is None:
            cls._live_options_params = set(inspect.signature(LiveOptions.__init__).parameters) - {
                "self"
            }
        return cls._live_options_params

    def apply_update(self: _S, delta: _S) -> Dict[str, Any]:
        """Merge a delta into this store, with delta-merge for ``live_options``.

        ``live_options`` is merged field-by-field (non-None fields from the
        delta overwrite corresponding fields in the stored options) rather
        than being replaced wholesale.

        ``model`` and ``language`` are kept in sync bidirectionally between
        the top-level settings fields and ``live_options``.
        """
        # Pull live_options out of the delta so super() doesn't replace it.
        delta_lo = getattr(delta, "live_options", NOT_GIVEN)
        if is_given(delta_lo):
            delta.live_options = NOT_GIVEN  # type: ignore[assignment]

        # Let the base class handle model, language, extra.
        changed = super().apply_update(delta)

        # Sync top-level model/language changes into stored live_options.
        if "model" in changed:
            self.live_options.model = self.model  # type: ignore[union-attr]
        if "language" in changed:
            self.live_options.language = self.language  # type: ignore[union-attr]

        # Merge live_options delta.
        if is_given(delta_lo):
            old_dict = self.live_options.to_dict()  # type: ignore[union-attr]
            delta_dict = delta_lo.to_dict()

            if delta_dict:
                merged = {**old_dict, **delta_dict}
                self.live_options = LiveOptions(**merged)

                for key in delta_dict:
                    old_val = old_dict.get(key, NOT_GIVEN)
                    if old_val != delta_dict[key]:
                        changed[key] = old_val

                # Sync model/language from live_options delta to top-level.
                if "model" in delta_dict and delta_dict["model"] != self.model:
                    changed.setdefault("model", self.model)
                    self.model = delta_dict["model"]
                if "language" in delta_dict and delta_dict["language"] != self.language:
                    changed.setdefault("language", self.language)
                    self.language = delta_dict["language"]

        return changed

    @classmethod
    def from_mapping(cls: Type[_S], settings: Mapping[str, Any]) -> _S:
        """Build a delta from a plain dict, routing LiveOptions keys correctly.

        Keys that are valid ``LiveOptions.__init__`` parameters (and not
        top-level ``STTSettings`` fields like ``model`` / ``language``) are
        collected into a ``LiveOptions`` object.  ``model`` and ``language``
        are routed to the top-level settings fields.  Truly unknown keys go
        to ``extra``.
        """
        lo_params = cls._get_live_options_params()
        stt_field_names = {"model", "language"}

        kwargs: Dict[str, Any] = {}
        lo_kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}

        for key, value in settings.items():
            canonical = cls._aliases.get(key, key)
            if canonical in stt_field_names:
                kwargs[canonical] = value
            elif canonical in lo_params:
                lo_kwargs[canonical] = value
            else:
                extra[key] = value

        if lo_kwargs:
            kwargs["live_options"] = LiveOptions(**lo_kwargs)

        instance = cls(**kwargs)
        instance.extra = extra
        return instance


class DeepgramSageMakerSTTService(STTService):
    """Deepgram speech-to-text service for AWS SageMaker.

    Provides real-time speech recognition using Deepgram models deployed on
    AWS SageMaker endpoints. Uses HTTP/2 bidirectional streaming for low-latency
    transcription with support for interim results, speaker diarization, and
    multiple languages.

    Requirements:

    - AWS credentials configured (via environment variables, AWS CLI, or instance metadata)
    - A deployed SageMaker endpoint with Deepgram model: https://developers.deepgram.com/docs/deploy-amazon-sagemaker
    - Deepgram SDK for LiveOptions configuration

    Example::

        stt = DeepgramSageMakerSTTService(
            endpoint_name="my-deepgram-endpoint",
            region="us-east-2",
            live_options=LiveOptions(
                model="nova-3",
                language="en",
                interim_results=True,
                punctuate=True,
            ),
        )
    """

    _settings: DeepgramSageMakerSTTSettings

    def __init__(
        self,
        *,
        endpoint_name: str,
        region: str,
        sample_rate: Optional[int] = None,
        live_options: Optional[LiveOptions] = None,
        ttfs_p99_latency: Optional[float] = DEEPGRAM_SAGEMAKER_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Deepgram SageMaker STT service.

        Args:
            endpoint_name: Name of the SageMaker endpoint with Deepgram model
                deployed (e.g., "my-deepgram-nova-3-endpoint").
            region: AWS region where the endpoint is deployed (e.g., "us-east-2").
            sample_rate: Audio sample rate in Hz. If None, uses value from
                live_options or defaults to the value from StartFrame.
            live_options: Deepgram LiveOptions configuration. Treated as a
                delta from a set of sensible defaults — only the fields you
                set are overridden; all others keep their default values.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to the parent STTService.
        """
        sample_rate = sample_rate or (live_options.sample_rate if live_options else None)

        # Create default options similar to DeepgramSTTService
        default_options = LiveOptions(
            encoding="linear16",
            language=Language.EN,
            model="nova-3",
            channels=1,
            interim_results=True,
            punctuate=True,
        )

        # Merge with provided options
        merged_dict = default_options.to_dict()
        if live_options:
            default_model = default_options.model
            merged_dict.update(live_options.to_dict())
            # Handle the "None" string bug from deepgram-sdk
            if "model" in merged_dict and merged_dict["model"] == "None":
                merged_dict["model"] = default_model

        # Convert Language enum to string if needed
        if "language" in merged_dict and isinstance(merged_dict["language"], Language):
            merged_dict["language"] = merged_dict["language"].value

        # Sync model/language to top-level STTSettings fields
        model = merged_dict.get("model")
        language = merged_dict.get("language")

        settings = DeepgramSageMakerSTTSettings(
            model=model, language=language, live_options=LiveOptions(**merged_dict)
        )

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=settings,
            **kwargs,
        )

        self._endpoint_name = endpoint_name
        self._region = region

        self._client: Optional[SageMakerBidiClient] = None
        self._response_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Deepgram SageMaker service supports metrics generation.
        """
        return True

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and warn about unhandled changes."""
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        # TODO: someday we could reconnect here to apply updated settings.
        # Code might look something like the below:
        # await self._disconnect()
        # await self._connect()

        self._warn_unhandled_updated_settings(changed)

        return changed

    async def start(self, frame: StartFrame):
        """Start the Deepgram SageMaker STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Deepgram SageMaker STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Deepgram SageMaker STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to Deepgram for transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (transcription results come via BiDi stream callbacks).
        """
        if self._client and self._client.is_active:
            try:
                await self._client.send_audio_chunk(audio)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
        yield None

    async def _connect(self):
        """Connect to the SageMaker endpoint and start the BiDi session.

        Builds the Deepgram query string from settings, creates the BiDi client,
        starts the streaming session, and launches background tasks for processing
        responses and sending KeepAlive messages.
        """
        logger.debug("Connecting to Deepgram on SageMaker...")

        live_options = LiveOptions(
            **{**self._settings.live_options.to_dict(), "sample_rate": self.sample_rate}
        )

        # Build query string from live_options, converting booleans to strings
        query_params = {}
        for key, value in live_options.to_dict().items():
            if value is not None:
                # Convert boolean values to lowercase strings for Deepgram API
                if isinstance(value, bool):
                    query_params[key] = str(value).lower()
                else:
                    query_params[key] = str(value)

        query_string = "&".join(f"{k}={v}" for k, v in query_params.items())

        # Create BiDi client
        self._client = SageMakerBidiClient(
            endpoint_name=self._endpoint_name,
            region=self._region,
            model_invocation_path="v1/listen",
            model_query_string=query_string,
        )

        try:
            # Start the session
            await self._client.start_session()

            # Start processing responses in the background
            self._response_task = self.create_task(self._process_responses())

            # Start keepalive task to maintain connection
            self._keepalive_task = self.create_task(self._send_keepalive())

            logger.debug("Connected to Deepgram on SageMaker")
            await self._call_event_handler("on_connected")

        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            await self._call_event_handler("on_connection_error", str(e))

    async def _disconnect(self):
        """Disconnect from the SageMaker endpoint.

        Sends a CloseStream message to Deepgram, cancels background tasks
        (KeepAlive and response processing), and closes the BiDi session.
        Safe to call multiple times.
        """
        if self._client and self._client.is_active:
            logger.debug("Disconnecting from Deepgram on SageMaker...")

            # Send CloseStream message to Deepgram
            try:
                await self._client.send_json({"type": "CloseStream"})
            except Exception as e:
                logger.warning(f"Failed to send CloseStream message: {e}")

            # Cancel keepalive task
            if self._keepalive_task and not self._keepalive_task.done():
                await self.cancel_task(self._keepalive_task)

            # Cancel response processing task
            if self._response_task and not self._response_task.done():
                await self.cancel_task(self._response_task)

            # Close the BiDi session
            await self._client.close_session()

            logger.debug("Disconnected from Deepgram on SageMaker")
            await self._call_event_handler("on_disconnected")

    async def _send_keepalive(self):
        """Send periodic KeepAlive messages to maintain the connection.

        Sends a KeepAlive JSON message to Deepgram every 5 seconds while the
        connection is active. This prevents the connection from timing out during
        periods of silence.
        """
        while self._client and self._client.is_active:
            await asyncio.sleep(5)
            if self._client and self._client.is_active:
                try:
                    await self._client.send_json({"type": "KeepAlive"})
                except Exception as e:
                    logger.warning(f"Failed to send KeepAlive: {e}")

    async def _process_responses(self):
        """Process streaming responses from Deepgram on SageMaker.

        Continuously receives responses from the BiDi stream, decodes the payload,
        parses JSON responses from Deepgram, and processes transcription results.
        Runs as a background task until the connection is closed or cancelled.
        """
        try:
            while self._client and self._client.is_active:
                result = await self._client.receive_response()

                if result is None:
                    break

                # Check if this is a PayloadPart with bytes
                if hasattr(result, "value") and hasattr(result.value, "bytes_"):
                    if result.value.bytes_:
                        response_data = result.value.bytes_.decode("utf-8")

                        try:
                            # Parse JSON response from Deepgram
                            parsed = json.loads(response_data)

                            # Extract and process transcript if available
                            if "channel" in parsed:
                                await self._handle_transcript_response(parsed)

                        except json.JSONDecodeError:
                            logger.warning(f"Non-JSON response: {response_data}")

        except asyncio.CancelledError:
            logger.debug("Response processor cancelled")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            logger.debug("Response processor stopped")

    async def _handle_transcript_response(self, parsed: dict):
        """Handle a transcript response from Deepgram.

        Extracts the transcript text, determines if it's final or interim, extracts
        language information, and pushes the appropriate frame (TranscriptionFrame
        or InterimTranscriptionFrame) downstream.

        Args:
            parsed: The parsed JSON response from Deepgram containing channel,
                alternatives, transcript, and metadata.
        """
        alternatives = parsed.get("channel", {}).get("alternatives", [])
        if not alternatives or not alternatives[0].get("transcript"):
            return

        transcript = alternatives[0]["transcript"]
        if not transcript.strip():
            return

        is_final = parsed.get("is_final", False)

        # Extract language if available
        language = None
        if alternatives[0].get("languages"):
            language = alternatives[0]["languages"][0]
            language = Language(language)

        if is_final:
            # Check if this response is from a finalize() call.
            # Only mark as finalized when both we requested it AND Deepgram confirms it.
            from_finalize = parsed.get("from_finalize", False)
            if from_finalize:
                self.confirm_finalize()
            await self.push_frame(
                TranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                    result=parsed,
                )
            )
            await self._handle_transcription(transcript, is_final, language)
            await self.stop_processing_metrics()
        else:
            # Interim transcription
            await self.push_frame(
                InterimTranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                    result=parsed,
                )
            )

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing.

        This method is decorated with @traced_stt for observability and tracing
        integration. The actual transcription processing is handled by the parent
        class and observers.

        Args:
            transcript: The transcribed text.
            is_final: Whether this is a final transcription result.
            language: The detected language of the transcription, if available.
        """
        pass

    async def _start_metrics(self):
        """Start processing metrics collection."""
        await self.start_processing_metrics()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with Deepgram SageMaker-specific handling.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        # Start metrics when user starts speaking (if VAD is not provided by Deepgram)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._start_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # https://developers.deepgram.com/docs/finalize
            # Mark that we're awaiting a from_finalize response
            self.request_finalize()
            if self._client and self._client.is_active:
                try:
                    await self._client.send_json({"type": "Finalize"})
                except Exception as e:
                    logger.warning(f"Error sending Finalize message: {e}")
            logger.trace(f"Triggered finalize event on: {frame.name=}, {direction=}")
