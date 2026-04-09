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

    def _add_aligned_mesh_override(self, cells_per_half_period=5):
        """
        Aligned mesh override for the shifted geometry (7-block layout).

        When delta == 0, delegates to the parent implementation (3 blocks).
        When delta > 0, uses 7 override boxes:
          1. Left periodic   (x_grating_start → L_narrow_1 start)  dx_grating (exact)
          2. Right periodic  (R_narrow_2 end  → x_grating_end)     dx_grating (exact)
          3. L_narrow_1      (hp - delta)                           snap_dx
          4. L_wide_1        (hp)                                   snap_dx
          5. Central         (cavity + R_narrow_1)                  snap_dx
          6. R_wide_1        (hp)                                   snap_dx
          7. R_narrow_2      (hp - delta)                           snap_dx

        Blocks 1 & 2 cover the purely periodic sections with exact dx_grating.
        Blocks 3-7 cover the non-periodic region with snap_dx, overlapping
        blocks 1 & 2 at the boundaries (Lumerical uses the finer dx).
        """
        delta = self.innermost_tooth_shift_m
        if delta == 0.0:
            super()._add_aligned_mesh_override(cells_per_half_period)
            return

        fdtd = self.fdtd
        half_pitch = 0.5 * self.pitch
        pitch = self.pitch
        n_cells_half = max(1, int(cells_per_half_period))
        dx_grating = half_pitch / float(n_cells_half)
        dy = self.width_narrow / 13.0
        dz = self.core_height / 7.0

        # Y/Z extent: waveguide + evanescent margin (1 tail for optimization, 2 for accurate)
        _dn_sq = max(self.n_eff_guess**2 - self.n_clad_const**2, 0.01)
        _decay_len = self.lambda_B / (2.0 * math.pi * math.sqrt(_dn_sq))
        _n_tails = 2.0 if self.simulation_mode == "accurate" else 1.0
        y_span_override = self.width_wide  + 2.0 * _n_tails * _decay_len
        z_span_override = self.core_height + 2.0 * _n_tails * _decay_len

        # --- Helper: snap_dx with ceil to ensure dx <= dx_grating ---
        def snap_dx_ceil(span):
            n = max(1, math.ceil(span / dx_grating))
            return span / float(n), n

        # --- Helper: add a mesh override box ---
        def add_mesh_box(name, x_left, x_right, dx_val):
            span = x_right - x_left
            fdtd.addmesh()
            fdtd.set("name", name)
            fdtd.set("x", x_left + span / 2.0)
            fdtd.set("x span", span)
            fdtd.set("y", 0.0)
            fdtd.set("y span", y_span_override)
            fdtd.set("z", 0.0)
            fdtd.set("z span", z_span_override)
            fdtd.set("override x mesh", 1)
            fdtd.set("override y mesh", 1)
            fdtd.set("override z mesh", 1)
            fdtd.set("dx", dx_val)
            fdtd.set("dy", dy)
            fdtd.set("dz", dz)

        # --- Tooth-edge positions (matching _add_bragg_core geometry) ---
        x_grating_start = -self.x_grating_end
        x_grating_end   =  self.x_grating_end
        actual_cavity_length = self.cavity_length + (2.0 * delta if self.lengthen_cavity else 0.0)
        shifted_span = half_pitch - delta

        # Left side
        x_L_narrow_1_start = x_grating_start + (self.n_periods_each_side - 1) * pitch
        x_L_wide_1_start   = x_L_narrow_1_start + shifted_span
        x_L_wide_1_end     = x_L_wide_1_start + half_pitch

        # Central: cavity + R_narrow_1
        x_cav_start    = x_L_wide_1_end
        central_span   = actual_cavity_length + half_pitch
        x_central_end  = x_cav_start + central_span

        # Right side
        x_R_wide_1_start  = x_central_end
        x_R_wide_1_end    = x_R_wide_1_start + half_pitch
        x_R_narrow_2_start = x_R_wide_1_end
        x_R_narrow_2_end   = x_R_narrow_2_start + shifted_span

        # Periodic spans = (n_periods - 1) * pitch → exact multiples of dx_grating
        n_left_per  = round((x_L_narrow_1_start - x_grating_start) / dx_grating)
        n_right_per = round((x_grating_end - x_R_narrow_2_end) / dx_grating)

        # Snap dx for non-periodic sections
        dx_shifted_narrow, n_shifted_narrow = snap_dx_ceil(shifted_span)
        dx_L_wide,   n_L_wide   = snap_dx_ceil(half_pitch)
        dx_central,  n_central  = snap_dx_ceil(central_span)
        dx_R_wide,   n_R_wide   = snap_dx_ceil(half_pitch)

        # Block 1: Left periodic (exact dx_grating)
        add_mesh_box("mesh_left_periodic", x_grating_start, x_L_narrow_1_start, dx_grating)

        # Block 2: Right periodic (exact dx_grating)
        add_mesh_box("mesh_right_periodic", x_R_narrow_2_end, x_grating_end, dx_grating)

        # Block 3: L_narrow_1
        add_mesh_box("mesh_L_narrow_1", x_L_narrow_1_start, x_L_wide_1_start, dx_shifted_narrow)

        # Block 4: L_wide_1
        add_mesh_box("mesh_L_wide_1", x_L_wide_1_start, x_L_wide_1_end, dx_L_wide)

        # Block 5: Central (cavity + R_narrow_1)
        add_mesh_box("mesh_central", x_cav_start, x_central_end, dx_central)

        # Block 6: R_wide_1
        add_mesh_box("mesh_R_wide_1", x_R_wide_1_start, x_R_wide_1_end, dx_R_wide)

        # Block 7: R_narrow_2
        add_mesh_box("mesh_R_narrow_2", x_R_narrow_2_start, x_R_narrow_2_end, dx_shifted_narrow)

        print(f"Mesh (shifted, 7-block): dx_grating={dx_grating*1e9:.1f}nm, delta={delta*1e9:.1f}nm")
        print(f"  1 Left periodic:  [{x_grating_start*1e6:.4f}, {x_L_narrow_1_start*1e6:.4f}] um  "
              f"dx={dx_grating*1e9:.1f}nm ({n_left_per} cells)")
        print(f"  2 Right periodic: [{x_R_narrow_2_end*1e6:.4f}, {x_grating_end*1e6:.4f}] um  "
              f"dx={dx_grating*1e9:.1f}nm ({n_right_per} cells)")
        print(f"  3 L_narrow_1:     [{x_L_narrow_1_start*1e6:.4f}, {x_L_wide_1_start*1e6:.4f}] um  "
              f"dx={dx_shifted_narrow*1e9:.1f}nm ({n_shifted_narrow} cells)")
        print(f"  4 L_wide_1:       [{x_L_wide_1_start*1e6:.4f}, {x_L_wide_1_end*1e6:.4f}] um  "
              f"dx={dx_L_wide*1e9:.1f}nm ({n_L_wide} cells)")
        print(f"  5 Central:        [{x_cav_start*1e6:.4f}, {x_central_end*1e6:.4f}] um  "
              f"dx={dx_central*1e9:.1f}nm ({n_central} cells)")
        print(f"  6 R_wide_1:       [{x_R_wide_1_start*1e6:.4f}, {x_R_wide_1_end*1e6:.4f}] um  "
              f"dx={dx_R_wide*1e9:.1f}nm ({n_R_wide} cells)")
        print(f"  7 R_narrow_2:     [{x_R_narrow_2_start*1e6:.4f}, {x_R_narrow_2_end*1e6:.4f}] um  "
              f"dx={dx_shifted_narrow*1e9:.1f}nm ({n_shifted_narrow} cells)")
        print(f"  Y/Z override: {y_span_override*1e6:.2f} x {z_span_override*1e6:.2f} um")

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
