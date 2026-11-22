"""Benchmark fairness and timing accounting, independent of GPU availability."""
from copy import deepcopy
import unittest

from experiments.wind_bubble.benchmark_tracking_workflow import check_run_consistency, summarize


def results():
    common = dict(source={'commit':'test', 'sha256':{}}, hostname='test-host', backend='cpu',
        device_kinds=['cpu'], cuda_visible_devices=None, preallocate='false', python='test', jax='test', numpy='test',
        num_cells=32, num_analysis=3, t_end=.02, requested_times=[0., .01, .02], solver_dt_max=.02/3,
        source_positions=[[.3,.3,.5],[.7,.3,.5],[.5,.7,.4]], source_strengths=[1.,1.,1.],
        discrete_energy_rates=[.3,.3,.3], source_budget_reference_sha256='reference')
    output = {}
    for mode, seconds in (('diagnostics',2.),('tracked',3.)):
        trial = dict(total_workflow_seconds=seconds, construction_seconds=.1,
            integration_seconds=seconds-.3, finalization_seconds=.2, primary_output_bytes=100,
            tracking_output_bytes=20 if mode=='tracked' else 0,
            output_payload_bytes=120 if mode=='tracked' else 100,
            iterations=5, final_state_sha256='same-state', final_time=.02, events=3)
        output[mode] = dict(deepcopy(common), mode=mode, compile_seconds=100., process_wall_seconds=200.,
            sampled_peak_gpu_process_mib=None, worker_peak_host_rss_bytes=1000,
            warmup=dict(trial,total_workflow_seconds=999.), trials=[dict(trial) for _ in range(3)])
    return output


class TrackingBenchmark(unittest.TestCase):
    def test_compilation_and_warmup_excluded_from_time_ratio(self):
        data = results()
        check_run_consistency(data)
        rows = summarize(data)
        self.assertEqual(rows[1]['median_workflow_seconds'], 3.)
        self.assertEqual(rows[1]['added_seconds_vs_diagnostics'], 1.)
        self.assertEqual(rows[1]['overhead_vs_diagnostics_percent'], 50.)
        self.assertEqual(rows[1]['compile_seconds'], 100.)
        self.assertEqual(rows[1]['median_finalization_seconds'], .2)
        self.assertEqual(rows[1]['median_tracking_output_bytes'], 20)
        self.assertIsNone(rows[1]['sampled_peak_gpu_process_mib'])

    def test_final_state_mismatch_rejects_comparison(self):
        data = results()
        data['tracked']['trials'][1]['final_state_sha256'] = 'changed-state'
        with self.assertRaisesRegex(AssertionError,'final states'):
            check_run_consistency(data)

    def test_bad_warmup_or_iteration_count_rejects_comparison(self):
        data = results()
        data['tracked']['warmup']['iterations'] = 6
        with self.assertRaisesRegex(AssertionError,'iteration counts'):
            check_run_consistency(data)

    def test_different_schedule_or_source_or_hardware_is_rejected(self):
        for key, value in (('requested_times',[0.,.005,.02]),('source_strengths',[1.,.5,1.]),
                           ('source',{'commit':'changed'}),('device_kinds',['other-GPU'])):
            data = results(); data['tracked'][key] = value
            with self.assertRaisesRegex(AssertionError,'settings, hardware'):
                check_run_consistency(data)

    def test_missing_events_or_repetitions_is_rejected(self):
        data = results(); data['tracked']['trials'][0]['events'] = 2
        with self.assertRaisesRegex(AssertionError,'event count'):
            check_run_consistency(data)
        data = results(); data['tracked']['trials'].pop()
        with self.assertRaisesRegex(AssertionError,'repetitions'):
            check_run_consistency(data)


if __name__ == '__main__':
    unittest.main()
