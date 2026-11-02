"""Independent geometry/interpolation checks for read-only ray diagnostics."""
import unittest
import jax.numpy as jnp
import numpy as np
from experiments.wind_bubble.inspect_reverse_sampling_paths import interpolate_rays, fractional_span


class SamplingPathTests(unittest.TestCase):
    def test_linear_field_includes_refined_origin_and_signed_normal(self):
        xyz=np.stack(np.meshgrid(*[np.arange(6)]*3,indexing='ij'))
        field=2*xyz[0]+3*xyz[1]-xyz[2]+10.
        base=np.array([[2.25,2.5,2.]])
        normal=np.array([[.6,.8,0.]])
        distance=np.array([[-.5,0,.75]])
        actual,valid=interpolate_rays(jnp.array(field[None]),jnp.array(base),jnp.array(normal),jnp.array(distance))
        coordinates=base[:,None,:]+distance[:,:,None]*normal[:,None,:]
        expected=2*coordinates[:,:,0]+3*coordinates[:,:,1]-coordinates[:,:,2]+10
        np.testing.assert_allclose(np.asarray(actual)[0],expected,rtol=1e-6)
        self.assertTrue(np.asarray(valid).all())

    def test_out_of_grid_is_missing_instead_of_clamped_evidence(self):
        values,valid=interpolate_rays(jnp.ones((1,4,4,4)),jnp.array([[.5,1.,1.]]),
                                     jnp.array([[1.,0.,0.]]),jnp.array([[-1.,0.,3.]]))
        np.testing.assert_array_equal(np.asarray(valid),[[False,True,False]])
        self.assertTrue(np.isnan(np.asarray(values)[0,0,[0,2]]).all())

    def test_one_cell_variation_is_relative_to_center_value(self):
        np.testing.assert_allclose(fractional_span(np.array([[8.,10.,12.],[3.,3.,3.]])),[.4,0.])

if __name__=='__main__': unittest.main()
