# Brief: antisymmetric TWIN pi-shift cavities (far-field cancellation by an antiphase copy) — 2026-10-24

For a NEW chat in this repo (branch `claude-te`). Read `CLAUDE.md` first (rules: propose runs before dispatch, ask Athena/IGUM,
smoke-test new builder code, short runs for high Q, no long validations unless asked).

## Idea (prior art: Rakich, Sandia, US 8,625,939 — https://patents.google.com/patent/US8625939B1/en ; not read in full yet)
Two identical resonators side by side, run in their ANTISYMMETRIC supermode (field in twin 2 = − field in twin 1). Every radiating
tooth then has an antiphase partner at separation s; their far fields cancel where the path difference k·s·u_y << 1.
For our device: two identical TE overshoot pi-shift gratings (Nt80x, 96 periods/side, 1619.4 nm) at y = ±s/2.

## Physics expectation (INFERENCE, to be tested)
- Overshoot teeth reach 1800 nm, so s_min (centre-to-centre) ≈ 1.9 µm + gap: k s ≈ 10 rad in-plane broadside -> no cancellation there.
- But ~40 % of the leak is GRAZING (|u_x| > 0.95, u_y small): path difference k s u_y small -> partial cancellation of the main leak.
- Strong evanescent coupling at small gaps splits symmetric/antisymmetric modes (Δλ); only the antisymmetric one benefits.
- Plain TE grating (1050-nm cavity, narrower teeth) allows smaller s -> stronger effect; test it too.

## Cheapest implementation (needs a builder knob — smoke-test + `debug_fsp_compare/scene_snapshot.py` regression)
Use the y = 0 MIRROR PLANE instead of building two devices: place ONE device at y = +s/2 and set the y-min boundary at y = 0 to the
parity that makes the mirror image the antiphase twin. Single-guide TE runs use `y min bc = Anti-Symmetric` (Ey even about y = 0);
the twin pair's antisymmetric supermode (Ey odd) should be `Symmetric`, the in-phase supermode `Anti-Symmetric` -> VERIFY the parity
in a smoke (Ex/Ey/Ez at y = 0 in a short build+run). Same numerics as the post-array study: dy29 core refinement
(`mesh.y_mesh_cells = 26`), box 6.8 um per HALF-domain (or larger: the device now sits at +s/2), short run 340 ps, far-field tube
(`farfield.save_surface_eh`, `short_run`), window ±0.75 nm (resonance shifts with s). Existing two-device code
(`bragg_device` n_devices=2, `runners/side_by_side/`, memory `project_side_by_side_coupling.md`) forces y-symmetry OFF and drives
only one guide — that is the OLD radiative-coupling study (gaps 1-2 um, passive twin), NOT this test.

## Test plan (propose to the user before dispatch)
1. Smoke: build + 5-ps run, check field parity at y = 0 for both BC choices.
2. Antisymmetric twin at 3 gaps (edge-to-edge 0.2 / 0.5 / 1.0 um) + the in-phase twin at the middle gap = 4 runs (~3.5 h each, A100).
3. Score: `python python_tools/radcancel/solve_v4.py --short --energy <copy of results_from_athena/budget_overshoot/dy29_box6p8.mat> <dev>`
   (Q_i from the ring-down; NOTE the stored energy now covers two cavities -> compare Q_i, not P/Pc). Bare single-device Q_i 5.67e5.
4. Report Q_i gain vs gap, Δλ, mode width; far-field band split (grazing vs broadside) to confirm the mechanism.
Decision: if the antisymmetric twin gains > ~20 % at a fab-realistic gap, it becomes a parallel track; it can also COMBINE with the
post arrays (best post device: Q_i x1.89, `nw_r2pack` in `runners/sweeps/postarray_overshoot_designs.json`).
