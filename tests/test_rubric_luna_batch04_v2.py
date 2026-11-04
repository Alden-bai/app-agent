"""Offline acceptance for the batch04 v2 writer after a zero-call v1 stall."""
import asyncio
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rubric_luna_batch04_production_v2 import Writer


class WriterTests(TestCase):
    def test_durable_ack_and_clean_shutdown(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / 'event'
            async def case():
                writer = Writer(maxsize=2)
                result = await asyncio.wait_for(writer.submit(lambda: target.write_text('committed')), 2)
                self.assertEqual(result, 9)
                self.assertEqual(target.read_text(), 'committed')
                await asyncio.wait_for(writer.close(), 2)
            asyncio.run(case())

    def test_failure_stops_further_writes(self):
        async def case():
            writer = Writer(maxsize=2)
            def fail():
                raise OSError('disk failure')
            with self.assertRaisesRegex(RuntimeError, 'disk failure'):
                await writer.submit(fail)
            with self.assertRaisesRegex(RuntimeError, 'writer_failed'):
                await writer.submit(lambda: None)
            await writer.close()
        asyncio.run(case())
