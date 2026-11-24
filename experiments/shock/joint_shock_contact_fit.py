"""Conditional inverse reconstruction of a planar shock/contact Euler fan.

The fitter receives only primitive cell values and spacing. Reference states,
positions, Mach numbers, and blur labels are not inputs. This is a restricted
model feasibility experiment, not a general shock finder or physical proof.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class FitSettings:
    gamma: float = 5 / 3
    blur_cells: tuple = (1, 3, 5)
    mach_upper: float = 20.
    starts: int = 6
    max_evaluations: int = 500
    fit_tolerance: float = 1.e-3
    score_noise_floor: float = 1.e-4
    score_margin: float = 10.
    rank_relative_tolerance: float = 1.e-7


def to_conserved(q, gamma):
    rho, velocity, pressure = np.asarray(q, float).T
    return np.column_stack([rho, rho*velocity, pressure/(gamma-1)+.5*rho*velocity**2])


def to_primitive(values, gamma):
    rho, momentum, energy = np.asarray(values, float).T
    velocity = momentum/rho
    return np.column_stack([rho, velocity, (gamma-1)*(energy-.5*momentum*velocity)])


def shock_states(rho, velocity, pressure, mach, gamma):
    """Right-going shock states, independent of the reference generator."""
    ratio = (gamma+1)*mach**2/((gamma-1)*mach**2+2)
    pressure_ratio = (2*gamma*mach**2-gamma+1)/(gamma+1)
    sound = np.sqrt(gamma*pressure/rho)
    downstream_velocity = velocity+mach*sound*(1-1/ratio)
    return np.array([rho, velocity, pressure]), np.array([rho*ratio, downstream_velocity, pressure*pressure_ratio])


def blur(values, width):
    if width == 1:
        return values.copy()
    padded = np.pad(values, ((width//2, width//2), (0, 0)), mode='edge')
    return np.column_stack([np.convolve(field, np.ones(width)/width, mode='valid') for field in padded.T])


def forward_fan(parameters, count, width=1, gamma=5/3, model='joint', velocity_scale=1.):
    """Conservative cell averages of a right-oriented, piecewise-constant fan.

    Joint parameters: log rho_up, u_up/scale, log P_up, Mach,
    log rho_contact, shock position in cell widths, shock-contact gap in cells.
    Contact can lie outside the observation window; the fit must then report
    loss of identifiability rather than inventing a measured contact position.
    """
    p = np.asarray(parameters, float)
    edges = np.arange(count+1, dtype=float)
    if model in ('joint', 'shock'):
        up, down = shock_states(np.exp(p[0]), p[1]*velocity_scale, np.exp(p[2]), p[3], gamma)
        if model == 'joint':
            contact = np.array([np.exp(p[4]), down[1], down[2]])
            shock_x, contact_x = p[5], p[5]-p[6]
            fc = np.clip(contact_x-edges[:-1], 0., 1.)
            fu = np.clip(edges[1:]-shock_x, 0., 1.)
            fd = np.maximum(1-fc-fu, 0.)
            values = np.column_stack([fc, fd, fu]) @ to_conserved(np.stack([contact, down, up]), gamma)
        else:
            fd = np.clip(p[4]-edges[:-1], 0., 1.)
            states = to_conserved(np.stack([down, up]), gamma)
            values = fd[:, None]*states[0]+(1-fd[:, None])*states[1]
    elif model == 'contact':
        left = np.array([np.exp(p[0]), p[2]*velocity_scale, np.exp(p[3])])
        right = np.array([np.exp(p[1]), left[1], left[2]])
        f = np.clip(p[4]-edges[:-1], 0., 1.)
        states = to_conserved(np.stack([left, right]), gamma)
        values = f[:, None]*states[0]+(1-f[:, None])*states[1]
    else:
        raise ValueError('Unknown forward model.')
    return blur(values, width)


def _context(profile, settings):
    q = np.asarray(profile, float)
    if (q.ndim != 2 or q.shape[1] != 3 or len(q) < 16 or not np.isfinite(q).all()
            or (q[:, (0, 2)] <= 0).any()):
        raise ValueError('Require at least 16 finite positive rho/u/P cell values.')
    if (settings.gamma <= 1 or settings.mach_upper <= 1 or settings.starts < 1
            or settings.fit_tolerance <= 0 or settings.score_noise_floor <= 0
            or any(not isinstance(w, (int, np.integer)) or w < 1 or w % 2 != 1 for w in settings.blur_cells)):
        raise ValueError('Invalid fit settings.')
    k = max(3, len(q)//8)
    # A data-derived Galilean frame keeps energy residuals from being dominated
    # by an arbitrary bulk velocity. No upstream/reference label is used.
    shift = .5*(np.median(q[:k, 1])+np.median(q[-k:, 1]))
    local = q.copy(); local[:, 1] -= shift
    values = to_conserved(local, settings.gamma)
    scale = np.maximum.reduce([np.ptp(values, axis=0), np.median(abs(values), axis=0), np.full(3, 1.e-6)])
    velocity_scale = np.sqrt(settings.gamma*np.median(local[:, 2])/np.median(local[:, 0]))
    return local, values, scale, float(shift), float(velocity_scale)


def _setup(q, velocity_scale, settings, model):
    n = len(q); k = max(3, n//8)
    left, right = np.median(q[:k], axis=0), np.median(q[-k:], axis=0)
    lr = (np.log(q[:, 0].min()*.01), np.log(q[:, 0].max()*100))
    lp = (np.log(q[:, 2].min()*.01), np.log(q[:, 2].max()*100))
    speed_limit = 5*np.sqrt(settings.gamma*q[:, 2]/q[:, 0]).max()
    vu = ((q[:, 1].min()-speed_limit)/velocity_scale, (q[:, 1].max()+speed_limit)/velocity_scale)
    gradient = abs(np.gradient(q[:, 2]))
    active = gradient > .25*gradient.max() if gradient.max() > 1.e-12 else np.zeros(n, bool)
    location = float(np.average(np.arange(n)[active]+.5, weights=gradient[active])) if active.any() else n/2
    ratio = max(left[2]/right[2], 1.)
    mach = np.clip(np.sqrt(((settings.gamma+1)*ratio+settings.gamma-1)/(2*settings.gamma)), 1.01, settings.mach_upper-.01)
    if model == 'joint':
        initial = [np.log(right[0]), right[1]/velocity_scale, np.log(right[2]), mach,
                   np.log(left[0]), location, 1.]
        lower = [lr[0], vu[0], lp[0], 1., lr[0], .5, 0.]
        upper = [lr[1], vu[1], lp[1], settings.mach_upper, lr[1], n-.5, n]
    elif model == 'shock':
        initial = [np.log(right[0]), right[1]/velocity_scale, np.log(right[2]), mach, location]
        lower = [lr[0], vu[0], lp[0], 1., .5]
        upper = [lr[1], vu[1], lp[1], settings.mach_upper, n-.5]
    else:
        initial = [np.log(left[0]), np.log(right[0]), np.mean(q[:, 1])/velocity_scale,
                   np.log(np.median(q[:, 2])), location]
        lower = [lr[0], lr[0], vu[0], lp[0], .5]
        upper = [lr[1], lr[1], vu[1], lp[1], n-.5]
    return np.array(initial), np.array(lower), np.array(upper)


def _score(residual, parameters, settings):
    count = residual.size
    mean_square = float(np.mean(residual**2))
    return float(count*np.log(max(mean_square, settings.score_noise_floor**2))+parameters*np.log(count))


def _fit_model(q, scale, velocity_scale, settings, model, width, fixed=None):
    data = to_conserved(q, settings.gamma)
    initial, lower, upper = _setup(q, velocity_scale, settings, model)
    free = np.ones(len(initial), bool)
    if fixed:
        for index, value in fixed.items():
            initial[index] = value; free[index] = False
    def unpack(z):
        full = initial.copy(); full[free] = z
        return full
    def residual(z):
        return ((forward_fan(unpack(z), len(q), width, settings.gamma, model, velocity_scale)-data)/scale).ravel()
    starts = []
    gaps = (0., .25, 1., 3., 8., len(q)/4)
    shifts = (0., -1., 1., -.5, .5, 0.)
    for index in range(settings.starts):
        seed = initial.copy()
        if model == 'joint':
            if free[6]: seed[6] = gaps[index % len(gaps)]
            if free[5]: seed[5] += shifts[index % len(shifts)]
        elif index:
            seed[-1] += shifts[index % len(shifts)]
        seed = np.minimum(np.maximum(seed, lower+1.e-10), upper-1.e-10)
        result = least_squares(residual, seed[free], bounds=(lower[free], upper[free]),
            x_scale='jac', max_nfev=settings.max_evaluations, ftol=1.e-11, xtol=1.e-11, gtol=1.e-11)
        full = unpack(result.x); errors = residual(result.x)
        singular = np.linalg.svd(result.jac, compute_uv=False)
        rank = int(np.sum(singular > settings.rank_relative_tolerance*singular[0])) if len(singular) and singular[0] else 0
        starts.append(dict(parameters=full.tolist(), rms=float(np.sqrt(np.mean(errors**2))),
            maximum_residual=float(abs(errors).max()), score=_score(errors, len(initial), settings),
            rank=rank, parameter_count=len(initial), success=bool(result.success), evaluations=int(result.nfev),
            smallest_relative_singular_value=float(singular[-1]/singular[0]) if len(singular) and singular[0] else 0.))
    starts.sort(key=lambda r:r['rms'])
    return dict(model=model, blur_cells=width, **starts[0], starts=starts)


def _fit_isentropic(q, scale, velocity_scale, settings):
    """A no-shock null with arbitrary observed pressure samples as nuisances.

    Constant entropy and one acoustic Riemann invariant constrain rho and u.
    N observed pressures count as N nuisance parameters in the BIC-style score;
    this is a declared conditional model comparison, not a calibrated posterior.
    """
    data = to_conserved(q, settings.gamma); pressure = q[:, 2]
    candidates = []
    for sign in (-1, 1):
        entropy = np.median(np.log(pressure)-settings.gamma*np.log(q[:, 0]))
        sound = np.sqrt(settings.gamma*pressure/q[:, 0])
        invariant = np.median(q[:, 1]-sign*2*sound/(settings.gamma-1))/velocity_scale
        def prediction(z):
            density = np.exp((np.log(pressure)-z[0])/settings.gamma)
            sound = np.sqrt(settings.gamma*pressure/density)
            velocity = z[1]*velocity_scale+sign*2*sound/(settings.gamma-1)
            return to_conserved(np.column_stack([density, velocity, pressure]), settings.gamma)
        def residual(z):
            return ((prediction(z)-data)/scale).ravel()
        result = least_squares(residual, [entropy, invariant], max_nfev=settings.max_evaluations,
                               ftol=1.e-11, xtol=1.e-11, gtol=1.e-11)
        errors = residual(result.x)
        candidates.append(dict(model='isentropic', blur_cells=None, parameters=result.x.tolist(),
            rms=float(np.sqrt(np.mean(errors**2))), maximum_residual=float(abs(errors).max()),
            score=_score(errors, len(q)+2, settings), rank=2, parameter_count=len(q)+2,
            success=bool(result.success), evaluations=int(result.nfev), wave_sign=sign))
    return min(candidates, key=lambda r:r['rms'])


def fit_profile(profile, grid_spacing, settings=FitSettings()):
    if not np.isfinite(grid_spacing) or grid_spacing <= 0:
        raise ValueError('Grid spacing must be finite and positive.')
    local, values, scale, shift, velocity_scale = _context(profile, settings)
    models = []
    for orientation in (1, -1):
        q = local.copy() if orientation == 1 else local[::-1]*np.array([1., -1., 1.])
        for model in ('joint', 'shock', 'contact'):
            for width in settings.blur_cells:
                fit = _fit_model(q, scale, velocity_scale, settings, model, width)
                fit.update(orientation=orientation)
                models.append(fit)
    null = _fit_isentropic(local, scale, velocity_scale, settings)
    null['orientation'] = 1
    models.append(null)
    by_model = []
    for name in ('joint', 'shock', 'contact', 'isentropic'):
        by_model.append(min((r for r in models if r['model'] == name),
                            key=lambda r:(r['score'], r['rms'])))
    by_model.sort(key=lambda r:r['score'])
    valid = [r for r in by_model if r['maximum_residual'] <= settings.fit_tolerance and r['success']]
    selected = valid[0]['model'] if valid else 'model_mismatch'
    margin = valid[1]['score']-valid[0]['score'] if len(valid) > 1 else None
    if len(valid) > 1 and margin < settings.score_margin:
        selected = 'ambiguous'
    joint = next(r for r in by_model if r['model'] == 'joint')
    p = np.array(joint['parameters']); sign = joint['orientation']; n = len(local)
    up, down = shock_states(np.exp(p[0]), p[1]*velocity_scale, np.exp(p[2]), p[3], settings.gamma)
    contact = np.array([np.exp(p[4]), down[1], down[2]])
    for state in (up, down, contact):
        state[1] = sign*state[1]+shift
    shock_x = (p[5] if sign == 1 else n-p[5])*grid_spacing
    contact_x = (p[5]-p[6] if sign == 1 else n-p[5]+p[6])*grid_spacing
    # Include every tested blur and orientation, not just restarts of the best
    # hypothesis. Equivalent fits with different latent states must stay visible.
    near = []
    for candidate in models:
        if candidate['model'] != 'joint':
            continue
        for restart in candidate['starts']:
            if restart['maximum_residual'] > settings.fit_tolerance or not restart['success']:
                continue
            z = restart['parameters']; direction = candidate['orientation']
            _, state = shock_states(np.exp(z[0]), z[1]*velocity_scale, np.exp(z[2]), z[3], settings.gamma)
            xs = z[5] if direction == 1 else n-z[5]
            xc = z[5]-z[6] if direction == 1 else n-z[5]+z[6]
            near.append(dict(orientation=direction, blur_cells=candidate['blur_cells'],
                mach=z[3], downstream_density=float(state[0]), shock_cells=xs,
                contact_cells=xc, gap_cells=z[6], rank=restart['rank']))
    spread_names = ('mach', 'downstream_density', 'shock_cells', 'contact_cells', 'gap_cells')
    spread = {name:float(np.ptp([r[name] for r in near])) for name in spread_names} if near else None
    supported = bool(selected == 'joint' and joint['rank'] == 7 and p[3] >= 1.3
                     and p[6] > .05 and 0 < contact_x < n*grid_spacing
                     and spread is not None and spread['mach'] < .05*p[3]
                     and spread['downstream_density'] < .05*down[0]
                     and max(spread['shock_cells'], spread['contact_cells'], spread['gap_cells']) < .25)
    return dict(selected_model=selected, score_margin_between_valid_models=margin,
        joint_reconstruction_supported=supported, local_full_rank=joint['rank'] == 7,
        recovered_mach=float(p[3]), recovered_shock_x=float(shock_x), recovered_contact_x=float(contact_x),
        recovered_gap_cells=float(p[6]), recovered_upstream=up.tolist(),
        recovered_downstream=down.tolist(), recovered_contact=contact.tolist(),
        joint_near_fit_parameter_spread=spread, joint_near_fits=near, galilean_shift=shift,
        velocity_scale=velocity_scale, conserved_scales=scale.tolist(), models=by_model,
        all_discrete_fits=models,
        identifiability_scope='Conditional on the fitted Euler-fan family, blur choices and parameter bounds. Local rank and multistart agreement do not establish uniqueness among all physical flows.')


def profile_parameter(profile, grid_spacing, fit, parameter, values, settings=FitSettings()):
    """Refit other joint parameters at fixed Mach or gap, without truth inputs."""
    if parameter not in ('mach', 'gap'):
        raise ValueError('Profile mach or gap.')
    local, _, scale, _, velocity_scale = _context(profile, settings)
    best = next(r for r in fit['models'] if r['model'] == 'joint')
    sign = best['orientation']
    q = local if sign == 1 else local[::-1]*np.array([1., -1., 1.])
    rows = []
    for value in values:
        candidate = _fit_model(q, scale, velocity_scale, replace_starts(settings, 2), 'joint',
                               best['blur_cells'], fixed={3 if parameter == 'mach' else 6:float(value)})
        rows.append(dict(parameter=parameter, fixed_value=float(value), rms=candidate['rms'],
                         maximum_residual=candidate['maximum_residual'], success=candidate['success']))
    return rows


def replace_starts(settings, starts):
    from dataclasses import replace
    return replace(settings, starts=starts)
