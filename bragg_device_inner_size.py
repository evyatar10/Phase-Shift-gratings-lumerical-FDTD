"""
Extension of PiShiftBraggFDTDWithShift with an optional override of the
innermost tooth corrugation depth.

When innermost_corrugation_depth_m is None (default), _add_bragg_core()
delegates entirely to the parent — no change in behavior.
When a value is provided, the innermost period (d=1) on each side of the
cavity is built with that corrugation depth instead of the global one.

Physical interpretation:
    - innermost_corrugation_depth_m = 0       → flat innermost tooth (no corrugation)
    - innermost_corrugation_depth_m = depth   → innermost tooth has depth nm corrugation
    - innermost_corrugation_depth_m = None    → same as all other periods (default)

The conceptual maximum equals the global corrugation_depth_m (e.g. 300 nm),
which is not hard-coded here — it lives in GeometryConfig.corrugation_depth_m.
"""

import numpy as np

from bragg_device_shifted import PiShiftBraggFDTDWithShift


class PiShiftBraggFDTDWithInnerSize(PiShiftBraggFDTDWithShift):
    """
    Subclass of PiShiftBraggFDTDWithShift that additionally supports an
    independent corrugation depth for the innermost grating tooth on each side.

    All constructor arguments are identical to the parent class, with one
    additional optional keyword:

        innermost_corrugation_depth_m (float or None): Corrugation depth [m]
            to use exclusively for the innermost period (d=1) on each side.
            None means "same as every other period" (default — no change).
            0.0 means a flat innermost tooth (no width modulation).
            Must be >= 0 when specified.

    Can be combined with innermost_tooth_shift_m from the parent class:
    the tooth is first sized by this parameter, then the position shift is
    applied independently.
    """

    def __init__(self, *args, innermost_corrugation_depth_m=None, **kwargs):
        super().__init__(*args, **kwargs)
        if innermost_corrugation_depth_m is None:
            self.innermost_corrugation_depth_m = None
        else:
            self.innermost_corrugation_depth_m = float(innermost_corrugation_depth_m)
            if self.innermost_corrugation_depth_m < 0:
                raise ValueError(
                    f"innermost_corrugation_depth_m must be >= 0. "
                    f"Got {self.innermost_corrugation_depth_m * 1e9:.1f} nm."
                )

    def _add_bragg_core(self):
        """
        If innermost_corrugation_depth_m is None, delegates entirely to the
        parent implementation (which handles shift and baseline geometry).
        Otherwise, applies the modified innermost-tooth widths on top of the
        shift geometry.
        """
        if self.innermost_corrugation_depth_m is None:
            super()._add_bragg_core()
            return

        # --- Modified innermost size + optional shift ---
        delta = self.innermost_tooth_shift_m
        half_pitch = self.pitch / 2.0

        if delta != 0.0 and not (0.0 < delta < half_pitch):
            raise ValueError(
                f"innermost_tooth_shift_m must be in (0, half_pitch). "
                f"Got {delta * 1e9:.1f} nm, half_pitch={half_pitch * 1e9:.1f} nm."
            )

        fdtd = self.fdtd
        z_core_center = 0.0
        seg_id = 0

        def add_core_segment(x1, x2, width, name_prefix="core_seg"):
            nonlocal seg_id
            seg_id += 1
            fdtd.addrect()
            fdtd.set("name", f"{name_prefix}_{seg_id:d}")
            fdtd.set("material", self.core_material)
            fdtd.set("y", 0)
            fdtd.set("y span", width)
            fdtd.set("z", z_core_center)
            fdtd.set("z span", self.core_height)
            fdtd.set("x min", x1)
            fdtd.set("x max", x2)

        pitch = self.pitch
        avg_width = 0.5 * (self.width_narrow + self.width_wide)
        full_depth_edge = self.width_wide - self.width_narrow
        full_depth_center = self.center_mod_depth if self.use_apodization else full_depth_edge
        n_total = self.n_periods_each_side
        n_apod = self.n_apod_periods_each_side
        apod_method = self.apod_method
        tanh_steepness = self.tanh_steepness

        def get_mod_depth(d):
            if d <= n_apod and n_total > 1:
                denom = (n_apod - 1) if (n_apod > 1 and n_apod == n_total) else n_apod
                if denom == 0:
                    return full_depth_center
                frac = (d - 1) / float(denom)
                if apod_method == 'tanh':
                    frac = np.tanh(tanh_steepness * 2.0 * frac) / np.tanh(2.0 * tanh_steepness)
                return full_depth_center + (full_depth_edge - full_depth_center) * frac
            else:
                return full_depth_edge

        W_narrow, W_wide = {}, {}
        for d in range(1, n_total + 1):
            mod_depth = get_mod_depth(d)
            delta_w = mod_depth / 2.0
            W_narrow[d] = avg_width - delta_w
            W_wide[d] = avg_width + delta_w

        # Override innermost (d=1) widths with the specified corrugation depth
        inner_half = self.innermost_corrugation_depth_m / 2.0
        W_narrow[1] = avg_width - inner_half
        W_wide[1] = avg_width + inner_half

        x_grating_start = -self.x_grating_end
        x = x_grating_start
        add_core_segment(-self.x_sim_boundary - 1e-6, x_grating_start,
                         self.width_port, name_prefix="wg_left_inf")

        # Left grating: outer periods d = n_total down to d = 2
        for d in range(n_total, 1, -1):
            add_core_segment(x, x + half_pitch, W_narrow[d], name_prefix=f"L_narrow_{d}")
            x += half_pitch
            add_core_segment(x, x + half_pitch, W_wide[d], name_prefix=f"L_wide_{d}")
            x += half_pitch

        # Left innermost period (d=1): narrow gap shortened by delta when shift > 0
        add_core_segment(x, x + half_pitch - delta, W_narrow[1], name_prefix="L_narrow_1")
        x += half_pitch - delta
        add_core_segment(x, x + half_pitch, W_wide[1], name_prefix="L_wide_1")
        x += half_pitch

        # Cavity: enlarged by 2*delta only when lengthen_cavity=True
        cavity_extra = 2 * delta if self.lengthen_cavity else 0.0
        add_core_segment(x, x + self.cavity_length + cavity_extra, W_narrow[1], name_prefix="cavity")
        x += self.cavity_length + cavity_extra

        # Right innermost period (d=1): adjacent narrow stays at full half_pitch
        add_core_segment(x, x + half_pitch, W_narrow[1], name_prefix="R_narrow_1")
        x += half_pitch
        add_core_segment(x, x + half_pitch, W_wide[1], name_prefix="R_wide_1")
        x += half_pitch

        # Right second-from-cavity period (d=2): narrow gap shortened by delta when shift > 0
        add_core_segment(x, x + half_pitch - delta, W_narrow[2], name_prefix="R_narrow_2")
        x += half_pitch - delta
        add_core_segment(x, x + half_pitch, W_wide[2], name_prefix="R_wide_2")
        x += half_pitch

        # Right grating: outer periods d = 3 up to d = n_total
        for d in range(3, n_total + 1):
            add_core_segment(x, x + half_pitch, W_narrow[d], name_prefix=f"R_narrow_{d}")
            x += half_pitch
            add_core_segment(x, x + half_pitch, W_wide[d], name_prefix=f"R_wide_{d}")
            x += half_pitch

        add_core_segment(x, self.x_sim_boundary + 1e-6,
                         self.width_port, name_prefix="wg_right_inf")
