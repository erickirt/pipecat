#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram speech-to-text service implementation."""

import inspect
from dataclasses import dataclass, field, fields
from typing import Any, AsyncGenerator, Dict, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven
from pipecat.services.stt_latency import DEEPGRAM_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from deepgram import (
        AsyncListenWebSocketClient,
        DeepgramClient,
        DeepgramClientOptions,
        ErrorResponse,
        LiveOptions,
        LiveResultResponse,
        LiveTranscriptionEvents,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Deepgram, you need to `pip install pipecat-ai[deepgram]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class DeepgramSTTSettings(STTSettings):
    """Settings for the Deepgram STT service.

    Some commonly used ``LiveOptions`` fields are declared as top-level
    fields here so they can be updated individually via
    ``STTUpdateSettingsFrame``.  Any *additional* ``LiveOptions`` fields
    (e.g. ``filler_words``, ``diarize``, ``utterance_end_ms``) can be
    passed through the ``extra`` dict — they will be forwarded to
    ``LiveOptions`` when the WebSocket connection is (re)established.
    This keeps the settings class future-proof: new Deepgram features work
    without code changes on the Pipecat side.

    Parameters:
        encoding: Audio encoding format (e.g. ``"linear16"``).
        channels: Number of audio channels.
        interim_results: Whether to return interim transcription results.
        smart_format: Whether to enable Deepgram smart formatting.
        punctuate: Whether to add punctuation to transcripts.
        profanity_filter: Whether to filter profanity from transcripts.
        vad_events: Whether to enable Deepgram VAD events (deprecated).
    """

    encoding: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    channels: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    interim_results: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    smart_format: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    punctuate: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    profanity_filter: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    vad_events: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class DeepgramSTTService(STTService):
    """Deepgram speech-to-text service.

    Provides real-time speech recognition using Deepgram's WebSocket API.
    Supports configurable models, languages, and various audio processing options.

    Event handlers available (in addition to STTService events):

    - on_speech_started(service): Deepgram detected start of speech
    - on_utterance_end(service): Deepgram detected end of utterance

    Example::

        @stt.event_handler("on_speech_started")
        async def on_speech_started(service):
            ...
    """

    _settings: DeepgramSTTSettings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "",
        base_url: str = "",
        sample_rate: Optional[int] = None,
        live_options: Optional[LiveOptions] = None,
        addons: Optional[Dict] = None,
        should_interrupt: bool = True,
        ttfs_p99_latency: Optional[float] = DEEPGRAM_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Deepgram STT service.

        Args:
            api_key: Deepgram API key for authentication.
            url: Custom Deepgram API base URL.

                .. deprecated:: 0.0.64
                    Parameter `url` is deprecated, use `base_url` instead.

            base_url: Custom Deepgram API base URL.
            sample_rate: Audio sample rate. If None, uses default or live_options value.
            live_options: Deepgram LiveOptions for detailed configuration.
            addons: Additional Deepgram features to enable.
            should_interrupt: Determine whether the bot should be interrupted when Deepgram VAD events are enabled and the system detects that the user is speaking.

                .. deprecated:: 0.0.99
                    This parameter will be removed along with `vad_events` support.

            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to the parent STTService.

        Note:
            The `vad_events` option in LiveOptions is deprecated as of version 0.0.99 and will be removed in a future version. Please use the Silero VAD instead.
        """
        sample_rate = sample_rate or (live_options.sample_rate if live_options else None)

        if url:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'url' is deprecated, use 'base_url' instead.",
                    DeprecationWarning,
                )
            base_url = url

        default_options = LiveOptions(
            encoding="linear16",
            language=Language.EN,
            model="nova-3-general",
            channels=1,
            interim_results=True,
            smart_format=False,
            punctuate=True,
            profanity_filter=True,
            vad_events=False,
        )

        merged_options = default_options.to_dict()
        if live_options:
            default_model = default_options.model
            merged_options.update(live_options.to_dict())
            # NOTE(aleix): Fixes an in deepgram-sdk where `model` is initialized
            # to the string "None" instead of the value `None`.
            if "model" in merged_options and merged_options["model"] == "None":
                merged_options["model"] = default_model

        if "language" in merged_options and isinstance(merged_options["language"], Language):
            merged_options["language"] = merged_options["language"].value

        settings_fields = {f.name for f in fields(DeepgramSTTSettings)}
        settings_kwargs = {}
        extra = {}
        for key, value in merged_options.items():
            if key in settings_fields:
                settings_kwargs[key] = value
            else:
                extra[key] = value

        settings = DeepgramSTTSettings(**settings_kwargs)
        settings.extra = extra

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=settings,
            **kwargs,
        )

        self._addons = addons
        self._should_interrupt = should_interrupt

        if self._settings.vad_events:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "The 'vad_events' parameter is deprecated and will be removed in a future version. "
                    "Please use the Silero VAD instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        self._client = DeepgramClient(
            api_key,
            config=DeepgramClientOptions(
                url=base_url,
                options={"keepalive": "true"},  # verbose=logging.DEBUG
            ),
        )

        if self.vad_enabled:
            self._register_event_handler("on_speech_started")
            self._register_event_handler("on_utterance_end")

    @property
    def vad_enabled(self):
        """Check if Deepgram VAD events are enabled.

        Returns:
            True if VAD events are enabled in the current settings.
        """
        return self._settings.vad_events

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Deepgram service supports metrics generation.
        """
        return True

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if anything changed."""
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        await self._disconnect()
        await self._connect()

        return changed

    async def start(self, frame: StartFrame):
        """Start the Deepgram STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Deepgram STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Deepgram STT service.

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
            Frame: None (transcription results come via WebSocket callbacks).
        """
        await self._connection.send(audio)
        yield None

    def _build_live_options(self) -> LiveOptions:
        """Build a ``LiveOptions`` from flat settings fields, sample rate, and extras.

        Returns:
            A fully-populated ``LiveOptions`` ready for the Deepgram SDK.
        """
        valid_kwargs = set(inspect.signature(LiveOptions.__init__).parameters) - {"self"}

        # Start with extras that are valid LiveOptions kwargs.
        opts: dict[str, Any] = {k: v for k, v in self._settings.extra.items() if k in valid_kwargs}

        # Override with flat settings fields (these take precedence).
        s = self._settings
        opts.update(
            {
                "model": s.model,
                "language": s.language,
                "encoding": s.encoding,
                "channels": s.channels,
                "interim_results": s.interim_results,
                "smart_format": s.smart_format,
                "punctuate": s.punctuate,
                "profanity_filter": s.profanity_filter,
                "vad_events": s.vad_events,
                "sample_rate": self.sample_rate,
            }
        )

        return LiveOptions(**opts)

    async def _connect(self):
        logger.debug("Connecting to Deepgram")

        self._connection: AsyncListenWebSocketClient = self._client.listen.asyncwebsocket.v("1")

        self._connection.on(
            LiveTranscriptionEvents(LiveTranscriptionEvents.Transcript), self._on_message
        )
        self._connection.on(LiveTranscriptionEvents(LiveTranscriptionEvents.Error), self._on_error)

        if self.vad_enabled:
            self._connection.on(
                LiveTranscriptionEvents(LiveTranscriptionEvents.SpeechStarted),
                self._on_speech_started,
            )
            self._connection.on(
                LiveTranscriptionEvents(LiveTranscriptionEvents.UtteranceEnd),
                self._on_utterance_end,
            )

        if not await self._connection.start(
            options=self._build_live_options(), addons=self._addons
        ):
            await self.push_error(error_msg=f"Unable to connect to Deepgram")
        else:
            headers = {
                k: v
                for k, v in self._connection._socket.response.headers.items()
                if k.startswith("dg-")
            }
            logger.debug(f'{self}: Websocket connection initialized: {{"headers": {headers}}}')

    async def _disconnect(self):
        if await self._connection.is_connected():
            logger.debug("Disconnecting from Deepgram")
            # Deepgram swallows asyncio.CancelledError internally which prevents
            # proper cancellation propagation. This issue was found with
            # parallel pipelines where `CancelFrame` was not awaited for to
            # finish in all branches and it was pushed downstream reaching the
            # end of the pipeline, which caused `cleanup()` to be called while
            # Deepgram disconnection was still finishing and therefore
            # preventing the task cancellation that occurs during `cleanup()`.
            # GH issue: https://github.com/deepgram/deepgram-python-sdk/issues/570
            await self._connection.finish()

    async def _start_metrics(self):
        """Start processing metrics collection for this utterance."""
        await self.start_processing_metrics()

    async def _on_error(self, *args, **kwargs):
        error: ErrorResponse = kwargs["error"]
        logger.warning(f"{self} connection error, will retry: {error}")
        await self.push_error(error_msg=f"{error}")
        await self.stop_all_metrics()
        # NOTE(aleix): we don't disconnect (i.e. call finish on the connection)
        # because this triggers more errors internally in the Deepgram SDK. So,
        # we just forget about the previous connection and create a new one.
        await self._connect()

    async def _on_speech_started(self, *args, **kwargs):
        await self._start_metrics()
        await self._call_event_handler("on_speech_started", *args, **kwargs)
        await self.broadcast_frame(UserStartedSpeakingFrame)
        if self._should_interrupt:
            await self.push_interruption_task_frame_and_wait()

    async def _on_utterance_end(self, *args, **kwargs):
        await self._call_event_handler("on_utterance_end", *args, **kwargs)
        await self.broadcast_frame(UserStoppedSpeakingFrame)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _on_message(self, *args, **kwargs):
        result: LiveResultResponse = kwargs["result"]
        if len(result.channel.alternatives) == 0:
            return
        is_final = result.is_final
        transcript = result.channel.alternatives[0].transcript
        language = None
        if result.channel.alternatives[0].languages:
            language = result.channel.alternatives[0].languages[0]
            language = Language(language)
        if len(transcript) > 0:
            if is_final:
                # Check if this response is from a finalize() call.
                # Only mark as finalized when both we requested it AND Deepgram confirms it.
                from_finalize = getattr(result, "from_finalize", False)
                if from_finalize:
                    self.confirm_finalize()
                await self.push_frame(
                    TranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        result=result,
                    )
                )
                await self._handle_transcription(transcript, is_final, language)
                await self.stop_processing_metrics()
            else:
                # For interim transcriptions, just push the frame without tracing
                await self.push_frame(
                    InterimTranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        result=result,
                    )
                )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with Deepgram-specific handling.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame) and not self.vad_enabled:
            # Start metrics if Deepgram VAD is disabled & pipeline VAD has detected speech
            await self._start_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # https://developers.deepgram.com/docs/finalize
            # Mark that we're awaiting a from_finalize response
            self.request_finalize()
            await self._connection.finalize()
            logger.trace(f"Triggered finalize event on: {frame.name=}, {direction=}")
