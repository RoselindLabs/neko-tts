from copy import deepcopy
from unittest import IsolatedAsyncioTestCase
import base64
import hashlib

from audio import voice_message
from plugin import Config, NekoTTS


class TestPlugin(NekoTTS):
    @property
    def config(self):
        return self.test_config


class PrepareHistoryTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.p = TestPlugin()
        self.p.test_config = Config()
        self.p.config.plugin.enabled = self.p._ready = True
        self.now = 1000
        self.p._visibility.clock = lambda: self.now
        self.full_id = 'full-platform-voice-message-id'
        self.own = dict(handle='row/8', message_id=self.full_id, is_self=True, pure_voice=True)
        self.payload = dict(schema_version=1, session_id='chat-a', reply_message_id='target', entries=[self.own])
        self.message = dict(session_id='chat-a', message_id=self.full_id, raw_message=voice_message(b'clip')['segments'])
        self.p._visibility.remember_audio('chat-a', b'clip')

    async def assert_handles(self, expected, **kwargs):
        payload = dict(self.payload, **kwargs)
        original = deepcopy(payload)
        self.assertEqual(await self.p.prepare_history(**payload),
                         {'action': 'continue', 'custom_result': {'schema_version': 1, 'hide': expected}})
        self.assertEqual(payload, original)

    def test_sdk_registers_independent_blocking_hook_and_preserves_old_hook(self):
        hooks = [c['metadata'] for c in self.p.get_components() if c['type'] == 'HOOK_HANDLER']
        self.assertEqual([h['mode'] for h in hooks if h['hook'] == 'maisaka.replyer.prepare_history'], ['blocking'])
        old, = [h for h in hooks if h['hook'] == 'maisaka.replyer.before_model_request']
        self.assertNotIn('reply_history', old)

    async def test_only_confirmed_voice_not_prepared_audio_or_reply_tool_receipt(self):
        self.p._reply_visibility.remember('chat-a', 'call', 1, 'text_and_voice')
        for sent in (None, False, 1, 'partial'):
            await self.p.after_text_send(message=self.message, sent=sent)
            await self.assert_handles([])
        await self.p.after_text_send(message=dict(self.message, session_id='chat-b'), stream_id='chat-a', sent=True)
        await self.assert_handles([])
        await self.p.after_text_send(message=self.message, sent=True)
        await self.assert_handles(['row/8'])
        await self.assert_handles([], session_id='chat-b')
        await self.assert_handles([], reply_message_id=self.full_id)
        self.now += self.p._visibility.TTL
        await self.assert_handles([])

    async def test_flags_full_id_and_visible_rows_only(self):
        await self.p.after_text_send(message=self.message, sent=True)
        alias = 'm' + base64.b32encode(hashlib.sha1(self.full_id.encode()).digest()).decode().lower()[:6]
        rows = [self.own, dict(self.own, handle='second history occurrence')]
        for flag in ('is_self', 'pure_voice'):
            for value in (False, None, 1, 'true'):
                rows.append(dict(self.own, handle=f'{flag}-{value}', **{flag: value}))
        for i, wrong_id in enumerate([alias, self.full_id.upper(), self.full_id + ' ', '语音消息', [], None]):
            rows.append(dict(self.own, handle=f'wrong-{i}', message_id=wrong_id, text='[语音消息]', nickname='bot'))
        # Hook output is a projection over this round only, not a list of receipt IDs.
        await self.assert_handles(['row/8', 'second history occurrence'], entries=rows)
        await self.assert_handles([], entries=[])

    async def test_ambiguous_handles_and_quotes_are_preserved(self):
        await self.p.after_text_send(message=self.message, sent=True)
        quoted = dict(self.own, handle='quoting', message_id='user', is_self=False, quote=self.full_id)
        rows = [self.own, dict(quoted, handle=self.own['handle']), quoted, None,
                dict(self.own, handle=['bad']), dict(self.own, handle='unique')]
        await self.assert_handles(['unique'], entries=rows)

    async def test_missing_and_unknown_schema_and_not_ready_are_noops(self):
        await self.p.after_text_send(message=self.message, sent=True)
        for key in self.payload:
            payload = dict(self.payload)
            del payload[key]
            self.assertIsNone(await self.p.prepare_history(**payload))
        for fields in (dict(schema_version=True), dict(schema_version=1.0), dict(schema_version=2),
                       dict(session_id=[]), dict(reply_message_id=False), dict(entries={})):
            self.assertIsNone(await self.p.prepare_history(**dict(self.payload, **fields)))
        # Even cached receipts must not affect any prompt after disable/unload.
        items = [{'item_type': 'AssistantMessageItem', 'parts': [{'type': 'text', 'text': '[语音消息]'}]}]
        for enabled, ready in ((False, True), (True, False)):
            self.p.config.plugin.enabled, self.p._ready = enabled, ready
            self.assertIsNone(await self.p.prepare_history(**self.payload))
            self.assertIsNone(await self.p.hide_replyer_voice(items=items, session_id='chat-a'))
            self.assertIsNone(await self.p.hide_planner_voice(items=items, session_id='chat-a'))
