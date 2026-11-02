"""Catch scientific synchronization errors before an evolution animation renders."""
import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.wind_bubble.animate_sparse_evolution import load_events


class SparseEvolutionTests(unittest.TestCase):
    def make_case(self, directory):
        directory=Path(directory)
        rows=[]
        for i,values in enumerate([np.array([]),np.array([2.,4.])]):
            percentiles=np.percentile(values,[16,50,84]) if len(values) else [np.nan]*3
            rows.append(dict(time=i*.1,surface_cells_outside_injection=len(values),
                mach_valid_count=len(values),**dict(zip(('mach_p16','mach_median','mach_p84'),percentiles))))
            np.savez_compressed(directory/f'surfaces_{i:04d}.npz',schema_version=2,time=i*.1,
                positions=np.full((len(values),3),.5),mach=values,mach_valid=np.ones(len(values),bool))
        with (directory/'history.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        return directory

    def test_empty_initial_event_stays_missing_and_next_event_aligns(self):
        with tempfile.TemporaryDirectory() as tmp:
            events=load_events(self.make_case(tmp))
            self.assertEqual([e['time'] for e in events],[0.,.1])
            self.assertEqual(len(events[0]['mach']),0)
            self.assertTrue(np.isnan(float(events[0]['history']['mach_median'])))
            self.assertEqual(float(events[1]['history']['mach_median']),3.)

    def test_shifted_catalog_time_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_case(tmp);path=root/'surfaces_0001.npz'
            with np.load(path) as source: data={k:source[k] for k in source.files}
            data['time']=.2;np.savez_compressed(path,**data)
            with self.assertRaisesRegex(ValueError,'times differ'): load_events(root)

    def test_missing_event_and_wrong_population_summary_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_case(tmp)
            path=root/'surfaces_0001.npz'
            with np.load(path) as source: data={k:source[k] for k in source.files}
            data['mach']=np.array([20.,40.]);np.savez_compressed(path,**data)
            with self.assertRaises(AssertionError): load_events(root)
            path.unlink()
            with self.assertRaisesRegex(ValueError,'every history row'): load_events(root)

if __name__=='__main__': unittest.main()
