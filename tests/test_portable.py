"""Portable launch boundaries and actual cancellation-state regression."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from tools.models import entries,verify
from tools.run import command,environment,prepare_case
from tools.resident_client import wait_turn


class PortableTests(unittest.TestCase):
    def test_existing_pcm_preserved_at_an_unrelated_project_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'reference.jpg').write_bytes(b'image')
            audio=root/'audio.wav'
            with wave.open(str(audio),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(bytes(320000))
            original=audio.read_bytes()
            (root/'input.json').write_text(json.dumps([dict(reference_image='reference.jpg',audio='audio.wav',main_prompt='A person speaking.')]))
            path=prepare_case('input.json',root/'output',root)
            case=json.loads(path.read_text())[0]
            self.assertEqual(case['audio'],str(audio));self.assertEqual(audio.read_bytes(),original)
            self.assertTrue(Path(case['reference_image']).is_absolute())

    def test_short_audio_is_rejected_without_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'reference.jpg').write_bytes(b'image')
            with wave.open(str(root/'audio.wav'),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(bytes(32000))
            (root/'input.json').write_text(json.dumps([dict(reference_image='reference.jpg',audio='audio.wav')]))
            with self.assertRaisesRegex(ValueError,'10–15'):prepare_case('input.json',root/'output',root)

    def test_launch_keeps_hopper_settings_without_history_paths(self):
        root=Path('/tmp/another-developer/project')
        args=command('resident',root/'input.json',root)
        for pair in (('--nproc_per_node=2',),('--fps','20'),('--size','416*720'),('--audio_lookahead','0.5'),('--audio_history','4')):
            self.assertIn(pair[0],args)
            if len(pair)>1:self.assertEqual(args[args.index(pair[0])+1],pair[1])
        env=environment('resident',root/'run',root/'input.json',root)
        self.assertEqual(env['LIVEACT_H6_ROOT'],str(root/'run'))
        self.assertNotIn('zlu561',' '.join(args))
        self.assertEqual(env['HF_HUB_OFFLINE'],'1')

    def test_cancel_queue_state_is_not_final_render_completion(self):
        states=[dict(status='cancelled'),dict(status='cancelled',render=dict(status='cancelled',cancel_reason='deadline'))]
        with patch('tools.resident_client.request',side_effect=states),patch('tools.resident_client.time.sleep'):
            result=wait_turn(Path('/unused'),'turn')
        self.assertEqual(result['render']['cancel_reason'],'deadline')

    def test_exact_size_corrupt_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'weight').write_bytes(b'bad')
            row=dict(path='weight',size=3,sha256=hashlib.sha256(b'yes').hexdigest())
            with self.assertRaisesRegex(ValueError,'SHA256'):verify(row,root)

    def test_catalog_includes_all_ten_female_voices_once(self):
        rows=entries('speech');paths=[row['path'] for row in rows]
        self.assertEqual(len(paths),len(set(paths)))
        voices=[Path(path).stem for path in paths if '/voices/' in path]
        self.assertEqual(set(voices),{f'zf_{index:03d}' for index in (*range(1,9),17,18)})

    def test_verified_model_directory_symlink_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);project=root/'project';project.mkdir()
            shared=root/'shared';shared.mkdir();(shared/'weight').write_bytes(b'yes')
            (project/'models').symlink_to(shared,target_is_directory=True)
            row=dict(path='models/weight',size=3,sha256=hashlib.sha256(b'yes').hexdigest())
            self.assertTrue(verify(row,project)['expected_sha256_checked'])
            with self.assertRaisesRegex(ValueError,'escaped'):
                verify(dict(row,path='../shared/weight'),project)


if __name__=='__main__':unittest.main()
