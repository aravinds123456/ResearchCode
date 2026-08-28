from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branching_hallucinations.concurrency import bounded_map, clamp_concurrency


class ConcurrencyTests(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(clamp_concurrency(None), 8)
        self.assertEqual(clamp_concurrency(0), 1)
        self.assertEqual(clamp_concurrency(-3), 1)
        self.assertEqual(clamp_concurrency(4), 4)

    def test_bounded_map_preserves_order(self):
        async def _run():
            started = []

            async def _one(item: int) -> int:
                started.append(item)
                await asyncio.sleep(0.02 * (4 - item))
                return item * 10

            return await bounded_map([1, 2, 3], _one, concurrency=2), started

        results, started = asyncio.run(_run())
        self.assertEqual(results, [10, 20, 30])
        self.assertEqual(len(started), 3)
