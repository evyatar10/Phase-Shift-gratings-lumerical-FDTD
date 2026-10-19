"""Itai's Nt80 FWHM-20 overshoot (2026-10-18 delivery) as an N=98 TE device.

Study dir: runners/lumopt2_design/ | Created 2026-10-18 | Job(s): prod_confirm_te row 16
Purpose: geometry + base spec for his For_Evyatar_Nt80_FWHM20_optimize_er_dw.npy,
run by prod_confirm_te at the S2 numerics (conformal, fixed time) so it is
directly comparable to s2_seed_fixt (179082_5: T 0.98939, Q_L 7871).

GEOMETRY = his bragg_lib.apodized_bragg_resonator chain (share_with_Evyatar),
re-implemented without nazca and gated:
  * the same code on his OLD Nt60_highbulk_HighCorr_dw.npy with his IT15 flags
    reproduces runners/sweeps/itai_hh_apod.APOD_*_NM to 0.00 nm, cavity 950.28.
  * inputs (user + Itai 2026-10-18): N_t 80 (81 values), 98 periods/side ->
    17 bulk periods, bulk dw1 300 nm (npy[0] = 301.8), avg_wg 1.0 um, dw_middle 0.
  * advanced_index_correction ON: neff_LUT_2D.mat, per-period mid width solved so
    the period-averaged n_eff equals the bulk one (n_target from avg_wg +- dw1/2,
    bragg_lib:1225). target_lambda = 1560 nm (OUR operating lambda; his 1600
    moves no width by more than 0.8 nm).
  * advanced_dw_correction OFF (Itai: it is the sim->fab correction; for us
    everything is simulation). Drawn dw = npy dw to <=0.1 nm (round_gds).
The old S2 overshoot (itai_hh_nt60w20) DID carry the fab dw correction
(raw 484.9 -> drawn 494.2 bulk) and its mid widths match the chain with
n_target taken at the corrected bulk dw (exact), not bragg_lib's raw dw1 (+1.75 nm).

PITCH 489.65 nm: envelope-weighted <n> retune (itai_hh_nt60w20.pitch_for method)
referenced to S2 (491.06 nm -> measured 1559.86 nm): ratio 0.99713. EXPECTED
resonance 1559.86 +- ~0.6 nm (spread between the <n> and transfer-matrix ratios).
"""
import dataclasses

import numpy as np

from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import campaign_te_s2 as s2

N_SIDE, N_FREE = 98, 81                      # all N_t+1 = 81 apodized teeth free
PITCH_NM = 489.65
BULK_NARROW_NM, BULK_WIDE_NM = 850.0, 1150.0
CAVITY_W_NM = 981.3

# Drawn widths, cavity-first (d=1 .. d=81) = his .npy reversed through the chain above.
APOD_NARROW_NM = [
    981.3, 959.9, 942.5, 924.8, 908.1, 892.2, 877.0, 862.7,
    849.3, 836.7, 825.1, 814.4, 804.6, 795.8, 788.1, 781.4,
    775.8, 771.3, 767.9, 765.7, 764.8, 765.0, 766.6, 769.6,
    773.9, 779.2, 785.6, 793.2, 801.5, 810.2, 819.0, 827.7,
    835.8, 843.1, 849.2, 853.8, 856.9, 858.4, 858.3, 856.6,
    853.5, 849.4, 844.3, 838.4, 831.9, 825.2, 818.3, 811.7,
    805.2, 799.0, 793.4, 788.4, 784.1, 780.4, 777.5, 775.3,
    774.1, 773.8, 774.6, 776.3, 778.9, 782.4, 786.7, 791.5,
    796.8, 802.4, 808.2, 814.0, 819.6, 824.9, 829.6, 833.8,
    837.4, 840.6, 843.2, 845.4, 847.0, 848.3, 849.1, 849.6,
    849.3,
]
APOD_WIDE_NM = [
    981.4, 1003.5, 1022.8, 1043.7, 1064.8, 1086.1, 1107.8, 1129.6,
    1151.2, 1172.8, 1193.8, 1214.2, 1233.9, 1252.5, 1269.6, 1285.1,
    1298.5, 1309.6, 1318.1, 1323.8, 1326.0, 1325.4, 1321.4, 1313.9,
    1303.2, 1290.4, 1275.3, 1258.3, 1240.5, 1222.6, 1205.4, 1188.9,
    1174.3, 1161.6, 1151.4, 1143.8, 1138.7, 1136.3, 1136.5, 1139.2,
    1144.2, 1151.0, 1159.5, 1169.7, 1181.2, 1193.5, 1206.6, 1219.6,
    1232.7, 1245.6, 1257.8, 1268.9, 1278.8, 1287.4, 1294.5, 1299.6,
    1302.7, 1303.4, 1301.4, 1297.2, 1290.9, 1282.7, 1272.9, 1262.1,
    1250.6, 1238.6, 1226.7, 1215.0, 1204.1, 1194.2, 1185.5, 1178.0,
    1171.4, 1165.9, 1161.4, 1157.7, 1155.0, 1152.8, 1151.5, 1150.7,
    1151.1,
]
assert len(APOD_NARROW_NM) == len(APOD_WIDE_NM) == N_FREE

# Base: S2's v3 spec, re-keyed to this device. The frozen outer 17 teeth are the
# bulk 850/1150 (d=81 = npy[0] = 301.8 nm is NOT bulk, hence 81 free teeth).
_base = dataclasses.replace(
    s2.SPEC_V3, label="itai_nt80", pitch_nm=PITCH_NM,
    corr0_nm=BULK_WIDE_NM - BULK_NARROW_NM, avg_w_nm=(BULK_WIDE_NM + BULK_NARROW_NM) / 2,
    kappa_per_um=0.0572 * 300.0 / 494.2,       # DERIVED: S2 bulk kappa scaled by dw
    n_free=N_FREE, n_periods_side=N_SIDE, seed_override=None,
    region_dx_nm=PITCH_NM / eng.CELLS_PER_PITCH)


def seed_vector():
    """His 81 apodized teeth as (corr, avg) + zero shifts + inert comb + cavity."""
    nar = np.asarray(APOD_NARROW_NM, float)
    wid = np.asarray(APOD_WIDE_NM, float)
    p = eng.seed_params(dataclasses.replace(_base, corr_seed_nm=None, avg_seed_nm=None))
    L = eng.layout(N_FREE)
    p[L.SL_CORR] = wid - nar
    p[L.SL_AVG] = (wid + nar) / 2.0
    p[L.SL_SHIFT] = 0.0
    p[L.I_CAV] = CAVITY_W_NM
    return tuple(float(v) for v in p)


SEED = seed_vector()
SPEC = dataclasses.replace(_base, seed_override=SEED)


# ── OLD overshoot (S2 = Itai Nt60_FWHM_20) rebuilt with the index correction ONLY ──
# (user 2026-10-19; prod_confirm_te row 17). His raw .npy is lost (not on disk, Athena or
# transcripts), so the raw dw is recovered by inverting dw_correction_LUT.mat on the stored
# drawn tables (itai_hh_nt60w20); pushing it forward through both corrections returns the
# stored dw to 0.000 nm. Raw bulk 484.9 nm, peak 615.4. Then the same chain as above: dw corr
# OFF, index corr ON at 1560 nm, dw1 = 484.9, avg 1.0 um. vs stored S2: dw up to 18.9 nm
# smaller (the fab correction removed), mid widths -2.25..+1.15 nm. Pitch 490.98 nm (same
# <n> ratio method vs S2; it landed the Nt80 device at +0.05 nm).
S2IDX_N_FREE = 60
S2IDX_PITCH_NM = 490.98
S2IDX_BULK_NARROW_NM, S2IDX_BULK_WIDE_NM = 757.5, 1242.4
S2IDX_CAVITY_W_NM = 952.5
S2IDX_NARROW_NM = [
    952.5, 925.0, 915.7, 900.7, 884.0, 868.1, 853.2, 839.3,
    826.0, 813.5, 801.7, 790.7, 780.3, 770.8, 762.1, 754.3,
    747.2, 740.9, 735.2, 730.3, 726.4, 723.5, 721.5, 720.5,
    720.4, 721.3, 723.3, 726.3, 729.7, 733.8, 738.5, 743.7,
    749.1, 754.8, 760.4, 765.8, 771.1, 776.2, 781.0, 784.8,
    787.6, 789.3, 789.8, 789.3, 788.0, 786.1, 784.0, 781.9,
    779.8, 777.5, 775.2, 772.9, 770.6, 768.4, 766.4, 764.4,
    762.4, 760.5, 758.9, 757.9,
]
S2IDX_WIDE_NM = [
    952.5, 981.2, 991.6, 1009.1, 1029.6, 1050.5, 1071.2, 1091.8,
    1112.4, 1133.1, 1153.7, 1174.1, 1194.2, 1213.6, 1232.3, 1249.9,
    1266.3, 1281.7, 1296.2, 1308.8, 1319.2, 1327.2, 1332.7, 1335.6,
    1335.8, 1333.2, 1327.8, 1319.7, 1310.3, 1299.6, 1287.7, 1275.0,
    1261.9, 1248.7, 1236.1, 1224.2, 1213.0, 1202.6, 1193.0, 1185.4,
    1179.9, 1176.8, 1175.9, 1176.7, 1179.2, 1183.0, 1187.0, 1191.0,
    1195.3, 1199.9, 1204.6, 1209.4, 1214.2, 1218.7, 1223.1, 1227.4,
    1231.7, 1235.8, 1239.3, 1241.7,
]
# d=61 (npy[0] = 484.91) draws 757.5/1242.5 vs the bulk 757.5/1242.4 -- 0.1 nm, left frozen.
assert len(S2IDX_NARROW_NM) == len(S2IDX_WIDE_NM) == S2IDX_N_FREE

_s2idx_base = dataclasses.replace(
    s2.SPEC_V3, label="itai_s2_idxonly", pitch_nm=S2IDX_PITCH_NM,
    corr0_nm=round(S2IDX_BULK_WIDE_NM - S2IDX_BULK_NARROW_NM, 1),
    avg_w_nm=(S2IDX_BULK_WIDE_NM + S2IDX_BULK_NARROW_NM) / 2,
    kappa_per_um=0.0572 * 484.9 / 494.2, seed_override=None,
    region_dx_nm=S2IDX_PITCH_NM / eng.CELLS_PER_PITCH)


def s2idx_seed_vector():
    nar, wid = np.asarray(S2IDX_NARROW_NM, float), np.asarray(S2IDX_WIDE_NM, float)
    p = eng.seed_params(dataclasses.replace(_s2idx_base, corr_seed_nm=None, avg_seed_nm=None))
    L = eng.layout(S2IDX_N_FREE)
    p[L.SL_CORR] = wid - nar
    p[L.SL_AVG] = (wid + nar) / 2.0
    p[L.SL_SHIFT] = 0.0
    p[L.I_CAV] = S2IDX_CAVITY_W_NM
    return tuple(float(v) for v in p)


S2IDX_SEED = s2idx_seed_vector()
S2IDX_SPEC = dataclasses.replace(_s2idx_base, seed_override=S2IDX_SEED)
