"""Benchmark accounting checks independent of GPU availability."""
import csv
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from experiments.wind_bubble.benchmark_online_analysis import (
    parse_gpu_rows, planned_memory, summarize, compare_outputs,
)


class OnlineBenchmarkTests(unittest.TestCase):
    def test_gpu_sampler_counts_only_its_worker(self):
        self.assertEqual(parse_gpu_rows('99, 8000\n42, 10\n42, 20\n', 42), 30)
        self.assertIsNone(parse_gpu_rows('99, 8000\n42, [N/A]\n42, nan\n', 42))
        self.assertEqual(parse_gpu_rows('42, 0\n', 42), 0)

    def test_compiler_memory_accounts_for_aliases(self):
        stats = SimpleNamespace(temp_size_in_bytes=100, argument_size_in_bytes=20,
                                output_size_in_bytes=40, alias_size_in_bytes=10)
        executable = SimpleNamespace(memory_analysis=lambda: stats)
        self.assertEqual(planned_memory(executable)['planned_total_bytes'], 150)
        self.assertIsNone(planned_memory(SimpleNamespace(memory_analysis=lambda: None)))

    def test_compilation_and_warmup_do_not_enter_execution_overhead(self):
        results = {}
        for mode, seconds in [('solver_only', 2.), ('online', 3.), ('stored_offline', 4.)]:
            trials = [dict(total_workflow_seconds=seconds, integration_seconds=seconds,
                           offline_analysis_seconds=0., snapshot_transfer_write_seconds=0.,
                           output_payload_bytes=100, compact_bytes=100, snapshot_file_bytes=0)]
            results[mode] = dict(trials=trials, compile_integration_seconds=100.,
                                 compile_analysis_seconds=200., sampled_peak_gpu_process_mib=None,
                                 worker_peak_host_rss_bytes=1000, warmup={'total_workflow_seconds': 9999.})
        rows = summarize(results)
        self.assertEqual(rows[1]['overhead_vs_scheduled_solver_percent'], 50.)
        self.assertEqual(rows[1]['median_workflow_seconds'], 3.)
        self.assertEqual(rows[1]['compile_seconds'], 300.)
        self.assertIsNone(rows[1]['sampled_peak_gpu_process_mib'])

    def test_product_comparison_checks_histograms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ['online', 'stored_offline']:
                directory = root/mode/'trial_00'/'compact'
                directory.mkdir(parents=True)
                with (directory/'history.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['time', 'mach_median'])
                    writer.writeheader()
                    writer.writerow({'time': 0., 'mach_median': 'nan'})
                np.savez(directory/'histograms_0000.npz', mach_counts=[0, 0],
                         density_jump_counts=[0], mach_edges=[0, 1, 2], density_jump_edges=[0, 1])
            self.assertTrue(compare_outputs(root, 1)['passed'])
            np.savez(root/'stored_offline/trial_00/compact/histograms_0000.npz',
                     mach_counts=[1, 0], density_jump_counts=[0],
                     mach_edges=[0, 1, 2], density_jump_edges=[0, 1])
            with self.assertRaises(AssertionError):
                compare_outputs(root, 1)


if __name__ == '__main__':
    unittest.main()
