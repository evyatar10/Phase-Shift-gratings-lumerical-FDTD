"""
Extension of PiShiftBraggFDTD with an optional innermost-tooth shift.

When innermost_tooth_shift_m is not provided (defaults to 0.0), _add_bragg_core()
delegates to the parent class exactly — no change in behavior whatsoever.
When a non-zero shift is provided, the innermost period on each side is placed
with the tooth shifted away from the cavity center by that amount.
"""

import math

import numpy as np

from bragg_device import PiShiftBraggFDTD


class PiShiftBraggFDTDWithShift(PiShiftBraggFDTD):
    """
    Subclass of PiShiftBraggFDTD that supports an optional shift of the
    innermost grating tooth on each side of the cavity.

    All constructor arguments are identical to the parent class, with two
    additional optional keywords:

        innermost_tooth_shift_m (float): Distance [m] to shift the innermost
            tooth away from the cavity center on each side. Default 0.0.
            Must satisfy 0 < delta < half_pitch when non-zero.

        lengthen_cavity (bool): Controls whether the cavity is enlarged to
            compensate for the tooth shortening. Default True.
            - True:  cavity grows by 2*delta so total device length equals
                     the non-shifted device.
            - False: cavity stays at cavity_length so total device length is
                     2*delta shorter than the non-shifted device.

    Physical effect (around cavity, left to right):
        Before:               ...[L_narrow_1: hp][L_wide_1: hp][cavity: cav_len][R_narrow_1: hp][R_wide_1: hp][R_narrow_2: hp]...
        After (lengthen=True): ...[L_narrow_1: hp-d][L_wide_1: hp][cavity: cav_len+2d][R_narrow_1: hp][R_wide_1: hp][R_narrow_2: hp-d]...
        After (lengthen=False):...[L_narrow_1: hp-d][L_wide_1: hp][cavity: cav_len ][R_narrow_1: hp][R_wide_1: hp][R_narrow_2: hp-d]...

    L_narrow_1 (left innermost) and R_narrow_2 (right second-from-cavity)
    each shrink by delta. All other periods unchanged.
    """

    def __init__(self, *args, innermost_tooth_shift_m=0.0, lengthen_cavity=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.innermost_tooth_shift_m = float(innermost_tooth_shift_m)
        self.lengthen_cavity = bool(lengthen_cavity)

    def _add_aligned_mesh_override(self, cells_per_half_period=5, max_cavity_dx=40e-9):
        """
        Aligned mesh override for the shifted geometry.

        When delta == 0, delegates to the parent implementation.
        When delta > 0, creates five mesh boxes so that the L_narrow_1 and
        R_narrow_2 tooth edges (which are shifted by delta) fall exactly on
        mesh cell boundaries in the propagation (x) direction.

        Box layout:
            mesh_left_arm_outer  : sim_left  → L_narrow_1 start  (dx_grating)
            mesh_L_narrow_1      : L_narrow_1 span (hp-delta)     (snapped dx)
            mesh_cavity          : actual cavity span              (snapped dx)
            mesh_R_narrow_2      : R_narrow_2 span (hp-delta)     (snapped dx)
            mesh_right_arm_outer : R_narrow_2 end  → sim_right    (dx_grating)
        """
        delta = self.innermost_tooth_shift_m
        if delta == 0.0:
            super()._add_aligned_mesh_override(cells_per_half_period, max_cavity_dx)
            return

        fdtd = self.fdtd
        half_pitch = 0.5 * self.pitch
        n_cells_half = max(1, int(cells_per_half_period))
        dx_grating = half_pitch / float(n_cells_half)
        dy_global = self.width_narrow / 13.0
        dz_global = self.core_height / 7.0
        max_device_width = max(self.width_port, self.width_wide, self.width_narrow)
        y_span_override = max_device_width * 1.2
        z_span_override = self.core_height
        x_sim_left  = -self.sim_x_span / 2.0 - 1e-6
        x_sim_right =  self.sim_x_span / 2.0 + 1e-6

        # Actual cavity span (accounts for lengthen_cavity)
        actual_cavity_length = self.cavity_length + (2.0 * delta if self.lengthen_cavity else 0.0)
        x_cav_left  = -actual_cavity_length / 2.0
        x_cav_right =  actual_cavity_length / 2.0

        # Left arm: L_narrow_1 boundary positions
        # L_wide_1 (span=hp) sits immediately left of cavity; L_narrow_1 (span=hp-delta) sits left of L_wide_1
        x_L_narrow_1_end   = x_cav_left - half_pitch
        x_L_narrow_1_start = x_L_narrow_1_end - (half_pitch - delta)

        # Right arm: R_narrow_2 boundary positions
        # R_narrow_1 (hp) + R_wide_1 (hp) sit immediately right of cavity; then R_narrow_2 (hp-delta)
        x_R_narrow_2_start = x_cav_right + half_pitch + half_pitch
        x_R_narrow_2_end   = x_R_narrow_2_start + (half_pitch - delta)

        # Snapped dx for the shortened (hp - delta) segments
        n_cells_shifted = max(1, round((half_pitch - delta) / dx_grating))
        dx_shifted = (half_pitch - delta) / float(n_cells_shifted)

        def add_mesh_box(name, x, x_span, dx_val):
            fdtd.addmesh()
            fdtd.set("name", name)
            fdtd.set("x", x)
            fdtd.set("x span", x_span)
            fdtd.set("y", 0.0)
            fdtd.set("y span", y_span_override)
            fdtd.set("z", 0.0)
            fdtd.set("z span", z_span_override)
            fdtd.set("override x mesh", 1)
            fdtd.set("override y mesh", 1)
            fdtd.set("override z mesh", 1)
            fdtd.set("dx", dx_val)
            fdtd.set("dy", dy_global)
            fdtd.set("dz", dz_global)

        # Left outer arm: sim boundary → L_narrow_1 start
        len_left_outer = x_L_narrow_1_start - x_sim_left
        add_mesh_box("mesh_left_arm_outer",
                     x_sim_left + len_left_outer / 2.0, len_left_outer, dx_grating)

        # L_narrow_1 segment: exactly aligned to the shifted tooth span
        add_mesh_box("mesh_L_narrow_1",
                     x_L_narrow_1_start + (half_pitch - delta) / 2.0,
                     half_pitch - delta, dx_shifted)

        # R_narrow_2 segment: exactly aligned to the shifted tooth span
        add_mesh_box("mesh_R_narrow_2",
                     x_R_narrow_2_start + (half_pitch - delta) / 2.0,
                     half_pitch - delta, dx_shifted)

        # Right outer arm: R_narrow_2 end → sim boundary
        len_right_outer = x_sim_right - x_R_narrow_2_end
        add_mesh_box("mesh_right_arm_outer",
                     x_R_narrow_2_end + len_right_outer / 2.0, len_right_outer, dx_grating)

        # Cavity mesh: snapped to the actual (possibly enlarged) cavity length
        if self.use_cavity_mesh_override:
            target_dx = min(dx_grating, max_cavity_dx)
            n_cells_cav = max(1, math.ceil(actual_cavity_length / target_dx))
            dx_cav_snapped = actual_cavity_length / float(n_cells_cav)
            add_mesh_box("mesh_cavity", 0.0, actual_cavity_length, dx_cav_snapped)

    def _add_bragg_core(self):
        """
        If shift is 0, delegates entirely to the parent implementation.
        If shift > 0, applies the modified innermost-tooth geometry.
        """
        delta = self.innermost_tooth_shift_m

        # --- Zero shift: original behavior, not a single line of new code runs ---
        if delta == 0.0:
            super()._add_bragg_core()
            return

        # --- Non-zero shift: modified geometry ---
        half_pitch = self.pitch / 2.0

        if not (0.0 < delta < half_pitch):
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

        x_grating_start = -self.x_grating_end
        x = x_grating_start
        add_core_segment(-self.x_sim_boundary - 1e-6, x_grating_start,
                         self.width_port, name_prefix="wg_left_inf")

        # Left grating: outer periods d = n_total down to d = 2 (identical to parent)
        for d in range(n_total, 1, -1):
            add_core_segment(x, x + half_pitch, W_narrow[d], name_prefix=f"L_narrow_{d}")
            x += half_pitch
            add_core_segment(x, x + half_pitch, W_wide[d], name_prefix=f"L_wide_{d}")
            x += half_pitch

        # Left innermost period (d = 1): narrow gap shortened by delta
        add_core_segment(x, x + half_pitch - delta, W_narrow[1], name_prefix="L_narrow_1")
        x += half_pitch - delta
        add_core_segment(x, x + half_pitch, W_wide[1], name_prefix="L_wide_1")
        x += half_pitch

        # Cavity: enlarged by 2*delta only when lengthen_cavity=True
        cavity_extra = 2 * delta if self.lengthen_cavity else 0.0
        add_core_segment(x, x + self.cavity_length + cavity_extra, W_narrow[1], name_prefix="cavity")
        x += self.cavity_length + cavity_extra

        # Right innermost period (d = 1): unchanged — adjacent narrow stays at full half_pitch
        add_core_segment(x, x + half_pitch, W_narrow[1], name_prefix="R_narrow_1")
        x += half_pitch
        add_core_segment(x, x + half_pitch, W_wide[1], name_prefix="R_wide_1")
        x += half_pitch

        # Right second-from-cavity period (d = 2): narrow gap shortened by delta
        add_core_segment(x, x + half_pitch - delta, W_narrow[2], name_prefix="R_narrow_2")
        x += half_pitch - delta
        add_core_segment(x, x + half_pitch, W_wide[2], name_prefix="R_wide_2")
        x += half_pitch

        # Right grating: outer periods d = 3 up to d = n_total (identical to parent)
        for d in range(3, n_total + 1):
            add_core_segment(x, x + half_pitch, W_narrow[d], name_prefix=f"R_narrow_{d}")
            x += half_pitch
            add_core_segment(x, x + half_pitch, W_wide[d], name_prefix=f"R_wide_{d}")
            x += half_pitch

        add_core_segment(x, self.x_sim_boundary + 1e-6,
                         self.width_port, name_prefix="wg_right_inf")
