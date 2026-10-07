"""Measured envelope-FWHM bounds shared by proposal and result validation.

The historical QP band is an inner target, not the physical width contract.
Existing specs retain their two-sided fractional band. An explicit hard band
can express a seed-width upper bound without changing historical campaigns.
"""
import math


def _band(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f'{name} must contain two width bounds')
    lo, hi = map(float, value)
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo < hi):
        raise ValueError(f'{name} must be finite, positive and increasing')
    return lo, hi


def hard_band(spec, rho_dn, rho_up):
    """Return measured bounds, or None for an explicitly unguarded smoke."""
    explicit = getattr(spec, 'fwhm_bounds_um', None)
    if explicit is not None:
        return _band(explicit, 'fwhm_bounds_um')
    anchor = getattr(spec, 'fwhm0_um', None)
    if anchor is None:
        return None
    return _band((rho_dn * float(anchor), rho_up * float(anchor)), 'anchor band')


def qp_band(spec, rho_dn, rho_up):
    """Return inner QP bounds and validate containment in measured bounds."""
    hard = hard_band(spec, rho_dn, rho_up)
    margin = float(spec.wgp_margin_um)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('wgp_margin_um must be finite and nonnegative')
    explicit = getattr(spec, 'wgp_v3_band_um', None)
    if explicit is not None:
        inner = _band(explicit, 'wgp_v3_band_um')
    else:
        base = hard or _band((rho_dn * float(spec.wgp_target_um),
                              rho_up * float(spec.wgp_target_um)), 'smoke band')
        inner = _band((base[0] + margin, base[1] - margin), 'inner width band')
    if hard is not None and not (hard[0] <= inner[0] < inner[1] <= hard[1]):
        raise ValueError('QP width band lies outside the measured hard band')
    return inner


def violation(width, band):
    """Distance outside a measured band; missing/nonfinite width is infinite."""
    if width is None or not math.isfinite(float(width)):
        return math.inf
    if band is None:
        return 0.0
    return max(0.0, band[0] - float(width), float(width) - band[1])
