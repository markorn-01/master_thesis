import unittest
import numpy as np
from experiments.wind_bubble.compare_two_bubble_resolutions import validate_pair,neighborhood_settings,check_distribution


def metadata(n):
    return dict(status='complete',num_cells=n,source_positions=[[.35,.5,.5],[.65,.5,.5]],
                injection_radius=.0625,wind_terminal_velocity=10.,ambient_density=1.,ambient_pressure=.01,
                gamma=5/3,domain=[0.,1.],requested_times=[0.,.01],actual_times=[0.,.01],
                discrete_mass_rates=[.0065,.0065],discrete_energy_rates=[.325,.325],
                injection_prescription='same test scheme',source_budget_plan=dict(budget_matched=n>64))


class ResolutionComparisonTests(unittest.TestCase):
    def test_matching_metadata_and_unequal_prescribed_budgets(self):
        a,b=metadata(64),metadata(128);validate_pair(a,b)
        b['discrete_energy_rates']=[.4,.4]
        with self.assertRaises(ValueError):validate_pair(a,b)

    def test_schedule_and_unrecorded_matching_rejected(self):
        for key,value in [('actual_times',[0.,.02]),('source_budget_plan',{}),('ambient_density',2.)]:
            b=metadata(128);b[key]=value
            with self.assertRaises(ValueError):validate_pair(metadata(64),b)

    def test_physical_and_cell_scale_distinguished(self):
        settings={mode:(radius,cutoff) for mode,radius,cutoff in neighborhood_settings(128)}
        self.assertEqual(settings['native_cells'],(1.5,20))
        self.assertEqual(settings['native_cells_scaled_count'],(1.5,80))
        self.assertEqual(settings['common_physical_radius'],(3.,80))
        self.assertEqual(settings['common_physical_radius'][0]/128,1.5/64)

    def test_histogram_range_accounting_and_final_edge(self):
        values=np.array([.5,1.,2.,3.,3.5,np.nan,0.]);edges=np.array([1.,2.,3.])
        row=dict(mach_valid_count='5',mach_invalid_count='2',mach_underflow='1',mach_overflow='1')
        row.update({f'mach_{key}':str(v) for key,v in zip(['p16','median','p84'],np.percentile(values[:5],[16,50,84]))})
        check_distribution(values,edges,np.array([1,2]),row,'mach')
        with self.assertRaises(AssertionError):check_distribution(values,edges,np.array([1,1]),row,'mach')


if __name__=='__main__':unittest.main()
