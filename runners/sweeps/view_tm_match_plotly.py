"""
Plotly version of the combined-transmission figure (TE@80 vs period-matched TM).

Exports a standalone, fully interactive HTML (hover tooltips, box/lasso zoom, pan,
toggle traces in the legend, export-to-PNG button) that opens in any browser with
no Python running — and pops it open automatically.

Run locally:
    python runners/sweeps/view_tm_match_plotly.py
    python runners/sweeps/view_tm_match_plotly.py --all       # overlay every TM N
    python runners/sweeps/view_tm_match_plotly.py --no-open    # just write the HTML
    python runners/sweeps/view_tm_match_plotly.py --dir <folder>

Default folder: results_from_athena/tm_match_bisect/results
Writes:         <folder>/combined_transmission_TE80_vs_TMmatched.html
"""

import argparse
import glob
import os
import re

import numpy as np
import scipy.io as sio
import plotly.graph_objects as go


def _load(path):
    d = sio.loadmat(path)
    wl = np.asarray(d["wl_nm"]).squeeze().astype(float)
    T = np.asarray(d["T"]).squeeze().astype(float)
    o = np.argsort(wl)
    wl, T = wl[o], T[o]
    lam = float(np.asarray(d["resonance_wavelength_nm"]).squeeze())
    fwhm = abs(float(np.asarray(d["spectral_fwhm_nm"]).squeeze()))  # stored sign-flipped
    Q = lam / fwhm if fwhm > 0 else float("nan")
    return wl, T, lam, fwhm, Q


def main():
    ap = argparse.ArgumentParser(description="Interactive Plotly TE@80 vs matched-TM viewer.")
    ap.add_argument("--dir", default=os.path.join("results_from_athena", "tm_match_bisect", "results"))
    ap.add_argument("--all", action="store_true", help="overlay every TM period count")
    ap.add_argument("--match-N", type=int, default=132, help="matched TM period count")
    ap.add_argument("--no-open", action="store_true", help="write HTML but do not open a browser")
    args = ap.parse_args()

    te_files = [p for p in glob.glob(os.path.join(args.dir, "result_*.mat"))
                if re.search(r"_te(_|\.)", os.path.basename(p)) and "summary" not in p]
    tm_files = [p for p in glob.glob(os.path.join(args.dir, "result_*.mat"))
                if re.search(r"_tm(_|\.)", os.path.basename(p)) and "summary" not in p]
    if not te_files or not tm_files:
        raise SystemExit(f"No TE/TM result_*.mat found in {args.dir}")

    fig = go.Figure()
    hover = "%{fullData.name}<br>λ = %{x:.3f} nm<br>T = %{y:.4f}<extra></extra>"

    wl, T, lam, fwhm, Q = _load(te_files[0])
    fig.add_trace(go.Scatter(x=wl, y=T, mode="lines", name=f"TE  N=80  (Q≈{Q:.0f})",
                             line=dict(color="#ff7f0e", width=2.2), hovertemplate=hover))

    tm_by_N = {}
    for p in tm_files:
        m = re.search(r"_?N(\d+)", os.path.basename(p))
        if m:
            tm_by_N[int(m.group(1))] = p

    if args.all:
        for n in sorted(tm_by_N):
            wl, T, lam, fwhm, Q = _load(tm_by_N[n])
            fig.add_trace(go.Scatter(x=wl, y=T, mode="lines", name=f"TM  N={n}  (Q≈{Q:.0f})",
                                     line=dict(width=1.6), hovertemplate=hover))
    else:
        p = tm_by_N.get(args.match_N) or tm_by_N[max(tm_by_N)]
        wl, T, lam, fwhm, Q = _load(p)
        fig.add_trace(go.Scatter(x=wl, y=T, mode="lines", name=f"TM  N={args.match_N}  (Q≈{Q:.0f})",
                                 line=dict(color="#1f77b4", width=2.2), hovertemplate=hover))

    fig.update_layout(
        title="Combined transmission: TE@80 vs period-matched TM",
        xaxis_title="Wavelength [nm]", yaxis_title="Transmission, T",
        hovermode="x unified", template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        width=1000, height=600,
    )
    fig.update_xaxes(showspikes=True, spikemode="across")

    out_html = os.path.join(args.dir, "combined_transmission_TE80_vs_TMmatched.html")
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[plotly] wrote {out_html}")
    if not args.no_open:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(out_html))


if __name__ == "__main__":
    main()
