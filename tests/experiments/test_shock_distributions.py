"""Histogram synchronization and sample conservation in the visualization."""
import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.wind_bubble.animate_shock_distributions import load_distribution_events


class DistributionAnimationTests(unittest.TestCase):
    def make_case(self,directory):
        directory=Path(directory)
        mach=np.array([2.,4.,np.nan,31.]);density=np.array([.8,8.,np.nan,9.])
        row=dict(time=.1,surface_cells_outside_injection=4)
        archive=dict(time=.1)
        for name,values,edges in [('mach',mach,np.array([0.,2.,4.])),
                                  ('density_jump',density,np.array([0.,1.,2.,8.]))]:
            valid=np.isfinite(values)&(values>0);accepted=values[valid]
            row.update({f'{name}_valid_count':int(valid.sum()),f'{name}_invalid_count':int((~valid).sum()),
                        f'{name}_underflow':int((accepted<edges[0]).sum()),f'{name}_overflow':int((accepted>edges[-1]).sum())})
            row.update(zip([f'{name}_{k}' for k in ('p16','median','p84')],np.percentile(accepted,[16,50,84])))
            archive[name+'_edges']=edges;archive[name+'_counts']=np.histogram(accepted,bins=edges)[0]
        np.savez_compressed(directory/'surfaces_0000.npz',schema_version=2,time=.1,positions=np.full((4,3),.5),
            mach=mach,mach_valid=np.isfinite(mach)&(mach>0),density_jump=density,
            density_jump_valid=np.isfinite(density)&(density>0))
        np.savez_compressed(directory/'histograms_0000.npz',**archive)
        with (directory/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        return directory

    def test_subunity_density_final_edge_invalid_and_overflow_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            event=load_distribution_events(self.make_case(tmp))[0]
            info=event['density_jump_histogram']
            np.testing.assert_array_equal(info['counts'],[1,0,1])
            self.assertEqual((info['in_range'],info['invalid'],info['underflow'],info['overflow']),(2,1,0,1))
            self.assertEqual(info['median'],8.)  # All valid values, including overflow.
            self.assertEqual(sum(info[k] for k in ('in_range','invalid','underflow','overflow')),4)

    def test_wrong_histogram_event_time_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_case(tmp);path=root/'histograms_0000.npz'
            with np.load(path) as source: data={k:source[k] for k in source.files}
            data['time']=.2;np.savez_compressed(path,**data)
            with self.assertRaisesRegex(ValueError,'times differ'): load_distribution_events(root)

    def test_changed_counts_and_missing_archives_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_case(tmp);path=root/'histograms_0000.npz'
            with np.load(path) as source: data={k:source[k] for k in source.files}
            data['density_jump_counts'][0]=0;np.savez_compressed(path,**data)
            with self.assertRaises(AssertionError): load_distribution_events(root)
            path.unlink()
            with self.assertRaisesRegex(ValueError,'per saved event'): load_distribution_events(root)

if __name__=='__main__': unittest.main()
