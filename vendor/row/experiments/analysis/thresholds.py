"""Single source of truth for every declared threshold, boundary, and
statistical constant consumed by the executable preregistration
(analysis/gates.py, analysis/make_figures_w6.py).

make_figures_w5.py is deliberately NOT retro-wired to this module: it is
frozen as the archival as-implemented wave-5 analysis so the archived
w5_gate_report.json stays byte-reproducible; the corrected wave-5
re-audit in gates.py supersedes it for inference and cites this module.
"""

# classical stability references (frozen-preconditioner / unclipped
# recurrences ONLY; never a verdict criterion for clipped AdamW)
BOUNDARY = {"adamw": 38.0, "sgdm": 38.0, "sgd": 2.0}

# flattening threshold on the median row-map Jacobian singular value,
# with the declared sensitivity sweep and layer aggregations
FLATTEN_THR = 0.1
FLATTEN_SWEEP = (0.05, 0.1, 0.2)
LAYER_AGGS = ("min", "median", "max")

# operator-invariance endpoint on the stochastic-generation order
# parameter m_gen(G_LM)
MGEN_THR = 0.01

# force-law window machinery
W5_WLEN = 250          # wave-5 declared window length (steps)
W6_WLEN = 125          # wave-6 declared window length (steps)
FLIP_OSC = 0.4         # oscillatory-window marker: sign-flip rate above
FLIP_DEAD = 0.2        # no-coherent-carrier kill level
CAPTURED_MIN = 0.05    # captured-power floor for W3
RHO_PASS = -0.3        # force-law correlation pass level (both correlates)
RHO_KILL = 0.15        # |rho| below this kills
P_PASS = 0.01          # permutation p-value pass level
MIN_WINDOWS = 30       # declared minimum pooled windows for a verdict
INSUFF_WINDOWS = 15    # below this the force law is "insufficient"

# uncertainty
BOOT_BLOCKLEN = 10
BOOT_N = 1000
BOOT_ALPHA = 0.05

# wave-6 effect thresholds
CONTRAST_RATIO = 2.0   # W2 sustained/quenched contrast
DEPTH_RATIO = 2.0      # W5 damp-vs-base depth ratio; W7 freeze depth match
ONSET_DELAY = 2.0      # W7 onset-delay factor for carrier suppression
SEED_MAJORITY = 2      # "at least 2 of 3 seed pairs"

# P8 deep-layer tilt criterion (declared in wave5_prereg P8')
TILT_FACTOR = 10.0
TILT_CKPT = "ckpt_1000.pt"

# ---- wave-7 avalanche statistics (wave7_prereg.md; every value below
# is calibrated by the disclosed exploratory pass, pivot plan 2
# Section 4, and frozen before any wave-7 verdict is computed) ----
AV_THR_MULT = 3.0          # primary event threshold: 3x running median
AV_THR_SWEEP = (2.0, 3.0, 5.0)
AV_MED_WIN = 101           # centered running-median window (steps)
AV_WIN_SWEEP = (51, 101, 201)
AV_DUR_MULT = 2.0          # lowered threshold for duration statistics
AV_MATCH_LAG = 2           # data-event exclusion lag (dilated intervals)
AV_COINC_LAG = 3           # cross-block coincidence lag
AV_MIN_EVENTS = 50         # tail-fit verdict floor (else "insufficient")
AV_MIN_SD_EVENTS = 30      # D>=2 event floor for the size-duration fit
AV_XMIN_GRID = 100         # max xmin candidates (log-spaced)
AV_GOF_BOOT = 1000         # semiparametric gof bootstraps (CSN)
AV_GOF_P = 0.1             # power law not rejected at p >= this
AV_LR_P = 0.05             # Vuong LR significance level (two-sided)
AV_ALPHA_TOL = 0.3         # common-exponent tolerance across seeds
AV_ALPHA_KILL = 0.6        # pairwise exponent gap that kills W7.1b
AV_SWEEP_TOL = 0.4         # exponent stability under the threshold sweep
AV_CV_MIN = 1.2            # interevent-CV clustering level (Poisson = 1)
AV_N_SURR = 100            # surrogates per family (ar1, phase)
AV_SURR_Q = 0.95           # surrogate envelope quantile
AV_GAMMA_FLOOR = 1.0       # size-duration gamma CI must exclude this
AV_OMORI_EARLY = 25        # aftershock early window (steps after main)
AV_OMORI_LATE = 100        # aftershock late-window end (steps)
AV_OMORI_RATIO = 1.5       # early/late rate ratio pass level
AV_N_MAIN = 5              # mainshocks per cell
AV_N_SHIFT = 200           # circular shifts for coincidence/Omori nulls
AV_COINC_RATIO = 2.0       # coincidence pass level (x chance)
AV_REROUTE_DELTA = 1.5     # pairwise-ratio change factor (rerouting)
AV_LIVE_RATE_MIN = 0.5     # frozen cells: live-block rate floor (x base)
AV_DESCENT_LEN = 500       # stationarity split: descent length (steps)
AV_STAT_THIRDS = 3         # constant-eta window split count
AV_STAT_DRIFT = 2.0        # stationarity drift kill factor across thirds
AV_MARGIN_SKEW = 1.0       # W7.3a pass: (q95-q50)/(q50-q05) below this
AV_MARGIN_SKEW_KILL = 1.5  # W7.3a kill level
AV_RELAX_PROBES = 5        # W7.3b relaxation horizon (cadence-100 probes)
AV_RELAX_FRAC = 0.8        # W7.3b relaxed-excursion fraction
AV_RATE_WIN = 1000         # event-rate window (steps)
AV_RATE_BOOT_BLOCK = 100   # block length for rate bootstraps (steps)

# wave-7 constant-eta cells
W7_CONST_LR = 1e-3         # constant rate (collapsed phase per w1 scan)
W7_CONST_LR_SUB = 3e-4     # sub-critical constant control
W7_CONST_LR_FALLBACK = 7.5e-4  # declared divergence contingency rate
W7_CONST_WARMUP = 250      # numerical-safety ramp, excluded from stats
W7_CONST_STAT_START = 2000 # first step of the constant-eta window
W7_DOFF_OFFSET = 416000    # data-order control offset (chunks)

# wave-8 declared constants (frozen with wave8_prereg.md; append-only)
W8_CONST_LR = 7.5e-4       # boundary-rate constant trio (W8.1/W8.6)
W8_CONST_LR_FALLBACK2 = 6e-4   # declared divergence contingency rate
W8_PULSE_STARTS = tuple(range(3000, 22000, 2000))  # 10 pulses (W8.3)
W8_PULSE_LEN = 8           # steps per pulse (archived pulse length)
W8_PULSE_FACTOR = 2.0      # pulse rate = factor x base constant rate
W8_RESP_WIN = 200          # post-pulse response window (steps)
W8_MIN_PULSES = 8          # clean-pulse floor (else "insufficient")
W8_BRANCH_LAG = 25         # same-block cluster gap for sigma_hat (W8.4)
W8_GAMMA_BAND = (1.0, 4.0)  # cross-cell size-duration consistency band
W8_FLOORS = (0.3, 0.55)    # anneal-floor scan cells (W8.2)
W8_HOLD = 6000             # plateau end for the hold cell (W8.2c)
W8_LATE_STEPS = 4000       # late-anneal cell length (W8.2d)
W8_LATE_LO = 8000          # late-window start for floor-chain rates
W8_WIDTH_HEADS = (2, 4, 8)  # width scan (d_model 128/256/512; W8.5)
W8_FSS_Q = 0.99            # cutoff quantile for the width comparison
W8_NULL_Q = 0.95           # placement/Poisson null envelope quantile
W8_N_PLACE = 200           # seeded placements for W8.3/W8.4 nulls
