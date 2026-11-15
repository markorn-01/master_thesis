import copy
import unittest
import numpy as np
from experiments.astrolink.surface_history import HistorySettings,analyze_surface_history


class SurfaceHistory(unittest.TestCase):
    def catalogs(self):
        # Five disconnected patches become one connected strip. One-frame group
        # labels are computed, not supplied as tracking truth or source labels.
        x=np.concatenate([np.arange(4)*.01+k*.06 for k in range(5)])
        earlier=np.array([[a,b,0.] for a in x for b in np.arange(5)*.01])
        later=np.array([[a,b,.01] for a in np.arange(28)*.01 for b in np.arange(5)*.01])
        return [dict(positions=p,normals=np.tile([0.,0.,1.],(len(p),1)),grid_spacing=.01) for p in [earlier,later]]

    def settings(self,cap):return HistorySettings(max_partners=cap,association_gates=(.014,.018))

    def test_baseline_is_default_and_expanded_is_explicit(self):
        self.assertEqual(HistorySettings().max_partners,3)
        data=self.catalogs();before=copy.deepcopy(data)
        baseline=analyze_surface_history(data,[0.,.01],self.settings(3))
        expanded=analyze_surface_history(data,[0.,.01],self.settings(5))
        self.assertFalse(baseline['experimental']);self.assertTrue(expanded['experimental'])
        self.assertEqual(len(expanded['graph']['nodes']),6)
        self.assertTrue(any(e['kind']=='combination' and len(e['parents'])==5 and e['status']=='shared' for e in expanded['graph']['events']))
        self.assertFalse(any(len(e['parents'])==5 for e in baseline['graph']['events']))
        for old,new in zip(before,data):np.testing.assert_array_equal(old['positions'],new['positions'])

    def test_empty_states_and_nonzero_start_time(self):
        catalogs=[dict(positions=np.empty((0,3)),normals=np.empty((0,3)),grid_spacing=.1) for _ in range(3)]
        result=analyze_surface_history(catalogs,[1.,2.,3.])
        self.assertEqual(result['graph']['nodes'],[])
        self.assertEqual(len(result['records']),4)

    def test_invalid_schedule_grid_and_settings_rejected(self):
        data=self.catalogs()
        for times in [[0.],[0.,0.],[0.,np.nan]]:
            with self.assertRaises(ValueError):analyze_surface_history(data,times)
        data[1]['grid_spacing']=.02
        with self.assertRaises(ValueError):analyze_surface_history(data,[0.,1.])
        for kwargs in [dict(max_partners=4),dict(max_partners=True),dict(association_gates=(.2,.1)),dict(min_group_samples=0)]:
            with self.assertRaises(ValueError):HistorySettings(**kwargs)
