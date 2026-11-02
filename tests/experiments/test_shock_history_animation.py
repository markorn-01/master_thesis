"""Protect missing measurements, energy origins and no-gap-bridging in plots."""
import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.wind_bubble.animate_shock_histories import load_histories,energy_segments


def write(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class HistoryAnimationTests(unittest.TestCase):
    def make_case(self,tmp):
        root=Path(tmp);history=[];energy=[]
        times=[0.,.1,.2,.3,.4]
        power={'forward':[np.nan,2.,np.nan,4.,6.],'reverse':[np.nan,np.nan,np.nan,1.,1.]}
        totals={'forward':[np.nan,0.,0.,0.,.5],'reverse':[np.nan,np.nan,np.nan,0.,.1]}
        for i,t in enumerate(times):
            h=dict(time=t);e=dict(time=t,event_index=i)
            available=[]
            for k in ('forward','reverse'):
                h[k+'_surface_area']=.5 if np.isfinite(power[k][i]) else np.nan
                h[k+'_dissipation_rate']=power[k][i]
                e[k+'_dissipation_rate']=power[k][i]
                e[k+'_integration_interval_valid']=i==4
                e[k+'_cumulative_dissipated_energy']=totals[k][i]
                e[k+'_first_power_detection_time']=(.1 if k=='forward' else .3) if np.isfinite(totals[k][i]) else np.nan
                e[k+'_integrated_duration']=.1 if i==4 else 0.
                if np.isfinite(totals[k][i]):available.append(totals[k][i])
            e['combined_cumulative_dissipated_energy']=sum(available) if available else np.nan
            history.append(h);energy.append(e)
        write(root/'history.csv',history);write(root/'energy.csv',energy)
        return root,history,energy

    def test_carried_energy_is_preserved_but_missing_interval_is_not_drawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,_,_=self.make_case(tmp);data=load_histories(root);band=data['bands']['forward']
            self.assertTrue(np.isnan(band['energy'][0]))
            self.assertEqual(band['energy'][2],0.)
            self.assertFalse(band['energy_point_valid'][2])
            segments,index=energy_segments(data['times'],band['energy'],band['valid_interval'])
            np.testing.assert_array_equal(index,[4])
            np.testing.assert_allclose(segments,[[[.3,0.],[.4,.5]]])
            self.assertEqual(band['first_power_time'][-1],.1)
            self.assertEqual(data['bands']['reverse']['first_power_time'][-1],.3)

    def test_changed_energy_or_false_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,_,energy=self.make_case(tmp)
            energy[3]['forward_integration_interval_valid']=True;write(root/'energy.csv',energy)
            with self.assertRaisesRegex(ValueError,'flag'):load_histories(root)
            energy[3]['forward_integration_interval_valid']=False
            energy[3]['forward_cumulative_dissipated_energy']=.2;write(root/'energy.csv',energy)
            with self.assertRaises(AssertionError):load_histories(root)

    def test_shifted_energy_time_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,_,energy=self.make_case(tmp)
            energy[2]['time']=.25;write(root/'energy.csv',energy)
            with self.assertRaisesRegex(ValueError,'times differ'):load_histories(root)

if __name__=='__main__':unittest.main()
