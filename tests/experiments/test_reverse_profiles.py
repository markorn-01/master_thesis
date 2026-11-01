"""Geometry checks for the inspection output, independent of a solver run."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from experiments.wind_bubble.check_reverse_profiles import extract, plot


class ReverseProfileGeometryTests(unittest.TestCase):
    def setup_case(self, detected=True):
        shape=(4,4,4)
        centers=np.stack(np.meshgrid(*[np.arange(4)*.25+.125]*3,indexing='ij'),-1)
        mask=np.zeros(shape,bool)
        mask[1,1,1]=detected
        normal=np.full((3,)+shape,1/np.sqrt(3))
        result=SimpleNamespace(shock_surface_cells=mask,shock_direction=normal,
                               shock_surface_offsets=np.full(shape,.2))
        one=np.ones(shape)
        samples=(2*one,one,3*one,one,one.astype(bool),.4*one,.75*one)
        analyze=lambda state:(result,2*one,None,samples)
        registry=SimpleNamespace(density_index=0,pressure_index=1,
                                 velocity_index=SimpleNamespace(x=2,y=3,z=4))
        state=np.stack([one,one,0*one,0*one,0*one])
        return state,analyze,registry,SimpleNamespace(geometric_centers=centers),SimpleNamespace(grid_spacing=.25)

    def test_sample_positions_use_refined_center_and_opposite_normal_signs(self):
        _,surface=extract(*self.setup_case())
        self.assertEqual(len(surface),1)
        row=surface[0]
        normal=np.ones(3)/np.sqrt(3)
        position=np.full(3,.375)+.25*.2*normal
        self.assertAlmostEqual(row['radius'],np.linalg.norm(position-.5))
        self.assertAlmostEqual(row['upstream_radius'],np.linalg.norm(position+.25*.75*normal-.5))
        self.assertAlmostEqual(row['downstream_radius'],np.linalg.norm(position-.25*.4*normal-.5))
        self.assertTrue(row['upstream_inside_injection'])
        self.assertFalse(row['downstream_inside_injection'])
        self.assertEqual((row['rho1'],row['rho2'],row['p1'],row['p2']),(1.,3.,1.,2.))

    def test_no_detections_still_produces_profiles(self):
        profile,surface=extract(*self.setup_case(False))
        self.assertEqual(surface,[])
        self.assertEqual(len(profile),2)
        self.assertTrue(all(np.isfinite(r['density']) for r in profile))

    def test_plot_uses_requested_resolution_in_title(self):
        import matplotlib.pyplot as plt
        profile, surface = extract(*self.setup_case(False))
        with patch('matplotlib.figure.Figure.savefig'), patch.object(plt, 'close') as close:
            plot([(t, profile, surface) for t in (.12, .15, .20)], 'unused.png', num_cells=128)
            figure = close.call_args.args[0]
            self.assertIn('128^3', figure._suptitle.get_text())
        plt.close(figure)


if __name__=='__main__': unittest.main()
