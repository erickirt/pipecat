#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Utility for stripping non-speakable characters and markdown formatting from text.

Both NvidiaSageMakerHTTPTTSService and NvidiaSageMakerWebsocketTTSService
use :func:`sanitize_text_for_tts` so the logic lives in one place.
"""

import re

# ---------------------------------------------------------------------------
# Emoji / symbol ranges
# ---------------------------------------------------------------------------

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # Emoticons
    "\U0001f300-\U0001f5ff"  # Misc Symbols and Pictographs
    "\U0001f680-\U0001f6ff"  # Transport and Map
    "\U0001f700-\U0001f77f"  # Alchemical Symbols
    "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
    "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
    "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
    "\U0001fa00-\U0001fa6f"  # Chess Symbols
    "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027b0"  # Dingbats
    "\U0001f1e0-\U0001f1ff"  # Flags (iOS)
    "]+",
    flags=re.UNICODE,
)


def sanitize_text_for_tts(text: str) -> str:
    """Remove emojis and markdown formatting that should not be spoken aloud.

    Transformations applied (in order):
    1. Fenced code blocks  (``` ... ```) → removed entirely
    2. Markdown headers    (# / ## / …)  → header text kept, # stripped
    3. Horizontal rules    (--- / *** / ___) → removed
    4. Table separator rows (|---|---|)  → removed
    5. Table data rows      (| a | b |)  → cells joined with commas
    6. Bold / italic markers (**x**, *x*, __x__, _x_) → text kept, markers stripped
    7. Blockquote markers   (> …)        → marker stripped, text kept
    8. Inline code backticks (`x`)       → backticks stripped, text kept
    9. Emojis                            → removed
    10. Curly quotes → straight quotes
        Em/en dashes → comma (natural pause, not a spoken symbol)
        Separator hyphens ( - ) → comma
        Unordered list bullets (^- , ^* ) → removed
        Remaining bare * and _ → removed
        Other non-speakable symbols → removed
    11. Collapse extra whitespace

    Args:
        text: Raw text, potentially containing markdown and/or emoji.

    Returns:
        Plain text suitable for speech synthesis.
    """
    # 1. Fenced code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)

    # 2. Markdown headers  (# Heading → Heading)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # 3. Horizontal rules (---, ***, ___ on their own line)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # 4. Table separator rows  |---|:---:|---|
    text = re.sub(r"^\s*\|[\s\-|:]+\|\s*$", "", text, flags=re.MULTILINE)

    # 5. Table data rows  | cell | cell | → cell, cell
    def _table_row_to_csv(m: re.Match) -> str:
        cells = [c.strip() for c in m.group(1).split("|")]
        return ", ".join(c for c in cells if c)

    text = re.sub(r"^\s*\|(.+)\|\s*$", _table_row_to_csv, text, flags=re.MULTILINE)

    # 6. Bold / italic  (**x**, *x*, __x__, _x_)
    #    Handle triple before double before single to avoid partial matches.
    text = re.sub(r"\*{3}([^*]+)\*{3}", r"\1", text)
    text = re.sub(r"\*{2}([^*]+)\*{2}", r"\1", text)
    text = re.sub(r"\*([^*\s][^*]*[^*\s]|\S)\*", r"\1", text)
    text = re.sub(r"_{3}([^_]+)_{3}", r"\1", text)
    text = re.sub(r"_{2}([^_]+)_{2}", r"\1", text)
    text = re.sub(r"_([^_\s][^_]*[^_\s]|\S)_", r"\1", text)

    # 7. Blockquote markers
    text = re.sub(r"^\s*>\s*", "", text, flags=re.MULTILINE)

    # 8. Inline code backticks
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # 9. Emojis
    text = _EMOJI_PATTERN.sub("", text)

    # 10. Typographic and non-speakable characters

    # Curly quotes → straight equivalents (speakable)
    text = text.replace("\u2018", "'")  # LEFT SINGLE QUOTATION MARK
    text = text.replace("\u2019", "'")  # RIGHT SINGLE QUOTATION MARK
    text = text.replace("\u201c", '"')  # LEFT DOUBLE QUOTATION MARK
    text = text.replace("\u201d", '"')  # RIGHT DOUBLE QUOTATION MARK

    # Em/en dashes → comma (they mark a pause, not a spoken symbol)
    text = text.replace("\u2014", ", ")  # EM DASH
    text = text.replace("\u2013", ", ")  # EN DASH

    # Hyphen used as a separator ( - ) → comma; keep word-hyphens (e.g. "well-known")
    text = re.sub(r"(?<= )- | -(?= )", ", ", text)

    # Unordered list bullets at line start (not caught by step 3)
    text = re.sub(r"^[\s]*[-*]\s+", "", text, flags=re.MULTILINE)

    # Remaining bare * and _ (e.g. orphaned markers)
    text = text.replace("*", "")
    text = text.replace("_", " ")

    # Other symbols that TTS engines typically misread or glitch on
    text = re.sub(r"[\\|<>{}[\]~^=+#@]", "", text)

    # 11. Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # no more than one blank line
    text = re.sub(r"[ \t]+", " ", text)  # collapse spaces/tabs
    # text = text.strip()

    return text
