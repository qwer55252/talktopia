"""Sentence completion, ASR ordering, and the last-sentence boundary."""

import asyncio
from types import SimpleNamespace

import pytest

from talktopia.full_duplex.audio import AudioFrame
from talktopia.full_duplex.speech_backends import WindowedASR


def sentence_frame(index, *, final=False):
    return AudioFrame(
        utterance_id="speech",
        source_agent="Alice",
        chunk_index=index,
        frame_index=0,
        pcm_s16le=(index + 1).to_bytes(2, "little") * 960,
        duration_ms=40,
        is_chunk_end=True,
        is_utterance_end=final,
    )


@pytest.mark.asyncio
async def test_sentence_asr_is_ordered_and_never_exposes_later_text():
    release_first = asyncio.Event()
    second_decoded = asyncio.Event()

    async def decode(pcm, rate):
        if len(pcm) > 1920:
            return "Whole utterance recognized separately."
        index = int.from_bytes(pcm[:2], "little") - 1
        if index == 0:
            await release_first.wait()
        if index == 1:
            second_decoded.set()
        return f"Sentence {index}."

    asr = WindowedASR(
        SimpleNamespace(decode=decode), decode_interval_ms=1000, window_ms=3000
    )
    try:
        await asr.start_utterance("speech", "Bob")
        for index in range(2):
            await asr.push_audio(sentence_frame(index))
            asr.finish_sentence("speech", index, has_next_sentence=True)
        # The next sentence can arrive before the first sentence's ASR is ready.
        await asyncio.wait_for(second_decoded.wait(), 1)
        assert asr._updates.empty()
        release_first.set()
        updates = asr.updates()
        first = await asyncio.wait_for(anext(updates), 1)
        second = await asyncio.wait_for(anext(updates), 1)
        assert first.text == "Sentence 0." and first.sentence_index == 0
        assert second.text == "Sentence 0. Sentence 1." and second.sentence_index == 1
        assert first.sentence_texts == {0: "Sentence 0."}
        assert first.has_next_sentence and second.has_next_sentence
        assert not first.is_final and not second.is_final

        await asr.push_audio(sentence_frame(2, final=True))
        asr.finish_sentence("speech", 2, has_next_sentence=False)
        final = await asyncio.wait_for(asr.finish_utterance("speech"), 1)
        assert final.is_final and final.has_next_sentence is False
        assert final.text == "Whole utterance recognized separately."
        assert final.sentence_texts == {i: f"Sentence {i}." for i in range(3)}
        assert asr._updates.empty()  # The last sentence does not offer a backchannel.
    finally:
        await asr.close()


@pytest.mark.asyncio
async def test_sentence_asr_finishing_after_utterance_end_does_not_offer_backchannel():
    release_asr = asyncio.Event()

    async def decode(pcm, rate):
        await release_asr.wait()
        return "Received speech."

    asr = WindowedASR(
        SimpleNamespace(decode=decode), decode_interval_ms=1000, window_ms=3000
    )
    try:
        await asr.start_utterance("speech", "Bob")
        await asr.push_audio(sentence_frame(0))
        asr.finish_sentence("speech", 0, has_next_sentence=True)
        await asr.push_audio(sentence_frame(1, final=True))
        asr.finish_sentence("speech", 1, has_next_sentence=False)
        release_asr.set()
        final = await asyncio.wait_for(asr.finish_utterance("speech"), 1)
        assert final.is_final and final.has_next_sentence is False
        assert final.sentence_texts == {0: "Received speech.", 1: "Received speech."}
        assert asr._updates.empty()
    finally:
        await asr.close()
