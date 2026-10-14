"""Cancellation, context acknowledgements and bounded stream failure modes."""
import threading
import time
import unittest

from dialogue.session import DialogueSession, TurnStopped

PCM = bytes(16000)


class ASR:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
    def transcribe(self, pcm):
        self.entered.set()
        self.release.wait(2)
        return dict(text='我的问题', language='zh')


class LLM:
    def __init__(self):
        self.requests, self.closed = [], 0
    def stream(self, messages, **kwargs):
        self.requests.append(messages)
        try:
            yield '这是回答。'
        finally:
            self.closed += 1


class TTS:
    voice_id = 'zf_001'
    def __init__(self):
        self.voices = []
        self.closed = 0
    def stream(self, text, *, voice_id):
        self.voices.append(voice_id)
        try:
            yield b'\x01\x00'*16000
        finally:
            self.closed += 1


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.asr, self.llm, self.tts = ASR(), LLM(), TTS()
        self.sessions = []
    def tearDown(self):
        self.asr.release.set()
        for session in self.sessions:
            self.assertTrue(session.close(timeout=2))
    def session(self, **kwargs):
        session = DialogueSession(self.asr, self.llm, self.tts, **kwargs)
        self.sessions.append(session)
        return session
    def drain(self, session, turn):
        events = []
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            event = session.read_event(turn)
            if event:
                events.append(event)
                if event['type'] in ('complete', 'error', 'cancelled', 'no_speech'):
                    return events
        self.fail('Turn did not terminate')

    def test_only_acknowledged_answers_enter_bounded_history(self):
        session = self.session(max_history_pairs=1)
        first = session.submit_audio(PCM)
        self.assertEqual(self.drain(session, first)[-1]['type'], 'complete')
        self.assertEqual(session.history(), [])
        with self.assertRaises(ValueError):
            session.acknowledge_playback(first, last_audio_sequence=999)
        session.acknowledge_playback(first, last_audio_sequence=first.audio_packets-1)
        second = session.submit_audio(PCM, voice_id='zf_002')
        self.drain(session, second)
        self.assertEqual([row['role'] for row in self.llm.requests[-1]],
                         ['system', 'user', 'assistant', 'user'])
        session.acknowledge_playback(second, last_audio_sequence=second.audio_packets-1)
        self.assertEqual(len(session.history()), 1)
        self.assertEqual(self.tts.voices[-1], 'zf_002')

    def test_unplayed_answer_is_excluded_and_old_ack_rejected(self):
        session = self.session()
        first = session.submit_audio(PCM)
        self.drain(session, first)
        second = session.submit_audio(PCM)
        self.drain(session, second)
        self.assertEqual(len(self.llm.requests[-1]), 2)
        with self.assertRaises(TurnStopped):
            session.acknowledge_playback(first, last_audio_sequence=first.audio_packets-1)

    def test_cancel_full_output_queue_purges_audio_and_releases_generator(self):
        session = self.session(queue_events=2)
        first = session.submit_audio(PCM)
        deadline = time.monotonic()+2
        while time.monotonic() < deadline:
            event = session.read_event(first)
            if event and event['type'] == 'audio':
                break
        else:
            self.fail('No real output packet before backpressure test')
        while first.queue_high_water < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(first.queue_high_water, 2)
        started = time.monotonic()
        session.cancel(first)
        self.assertLess(time.monotonic()-started, 0.2)
        self.assertEqual(first.events.qsize(), 0)
        self.assertEqual(session.read_event(first)['type'], 'cancelled')
        self.assertIsNone(session.read_event(first))
        self.assertTrue(first.work_done.wait(1))
        self.assertEqual(self.llm.closed, 1)
        self.assertEqual(self.tts.closed, 1)
        second = session.submit_audio(PCM)
        self.assertEqual(self.drain(session, second)[-1]['type'], 'complete')
        self.assertTrue(first.work_done.wait(1))

    def test_native_call_timeout_fails_control_before_backend_returns(self):
        self.asr.release.clear()
        session = self.session(turn_timeout=0.05)
        turn = session.submit_audio(PCM)
        self.assertTrue(self.asr.entered.wait(1))
        self.assertTrue(turn.stopped.wait(1))
        self.assertEqual(session.read_event(turn)['reason'], 'generation_or_consumer_timeout')
        self.assertFalse(turn.work_done.is_set())
        self.assertEqual(turn.events.qsize(), 0)
        self.asr.release.set()
        self.assertTrue(turn.work_done.wait(1))
        self.assertEqual(len(self.llm.requests), 0)

    def test_old_recording_timer_cannot_expire_sealed_input(self):
        self.asr.release.clear()
        session = self.session()
        turn = session.begin_input()
        recording_generation = turn.timer_generation
        session.append_pcm(turn, PCM, sequence=0)
        session.end_input(turn)
        session._expire(turn, 'recording_eos_timeout', recording_generation)
        self.assertFalse(turn.stopped.is_set())
        self.asr.release.set()
        self.assertEqual(self.drain(session, turn)[-1]['type'], 'complete')

    def test_cancel_during_native_asr_discards_late_result(self):
        self.asr.release.clear()
        session = self.session()
        first = session.submit_audio(PCM)
        self.assertTrue(self.asr.entered.wait(1))
        session.cancel(first)
        second = session.submit_audio(PCM)
        self.asr.release.set()
        self.drain(session, second)
        self.assertEqual(len(self.llm.requests), 1)
        self.assertEqual(session.read_event(first)['type'], 'cancelled')

    def test_missing_eos_and_missing_consumer_expire_and_clear_buffers(self):
        session = self.session(recording_timeout=0.05, turn_timeout=0.1, queue_events=2)
        first = session.begin_input()
        session.append_pcm(first, PCM, sequence=0)
        self.assertEqual(self.drain(session, first)[-1]['reason'], 'recording_eos_timeout')
        self.assertEqual(len(first.input_pcm), 0)
        second = session.submit_audio(PCM)
        self.assertTrue(second.stopped.wait(1))
        self.assertEqual(session.read_event(second)['reason'], 'generation_or_consumer_timeout')
        self.assertTrue(second.work_done.wait(1))

    def test_input_sequence_alignment_budget_and_duplicate_eos(self):
        session = self.session()
        for chunk, sequence, reason in [(b'x', 0, 'input_requires'), (PCM, 1, 'input_sequence'),
                                         (bytes(480002), 0, 'input_duration')]:
            turn = session.begin_input()
            with self.assertRaisesRegex(ValueError, reason):
                session.append_pcm(turn, chunk, sequence=sequence)
            self.assertEqual(len(turn.input_pcm), 0)
        turn = session.submit_audio(PCM)
        with self.assertRaises(ValueError):
            session.end_input(turn)
        self.drain(session, turn)

    def test_unacknowledged_playback_expires_without_committing_history(self):
        session = self.session(playback_timeout=0.05)
        turn = session.submit_audio(PCM)
        self.drain(session, turn)
        self.assertTrue(turn.stopped.wait(1))
        self.assertEqual(session.read_event(turn)['reason'], 'playback_ack_timeout')
        self.assertEqual(session.history(), [])

    def test_oversized_tts_reply_fails_and_clears_pending_output(self):
        class OversizedTTS(TTS):
            def stream(self, text, *, voice_id):
                yield b'\x01\x00'*240001
        self.tts = OversizedTTS()
        session = self.session()
        turn = session.submit_audio(PCM)
        event = self.drain(session, turn)[-1]
        self.assertEqual(event['type'], 'error')
        self.assertIn('15-second budget', event['reason'])
        self.assertEqual(session.history(), [])


if __name__ == '__main__':
    unittest.main()
