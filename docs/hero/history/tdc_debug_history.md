# TDC Debug History

Archive of resolved bugs, fixes, and failed approaches. Kept for reference.

## Bugs Fixed (2026-02-05)
- Bug 1: Delta_T_b sign was flipped (`t_b_current - t_b_delayed` -> `t_b_delayed - t_b_current`)
- Bug 2: p_EE history off-by-one — `_update_tde_history` moved from start to end of `compute()`,
  delayed_idx changed from `(idx - delay)` to `(idx - delay + 1)` so p_EE/omega/lf are same timestep

## Lambda_inv DLS (2026-02-05)
- Replaced 1/lf with adaptive DLS: `dls_factor = lf / (lf^2 + lambda^2)` where `lambda^2 = 25*(1 - lf/F_bu)`
- At 0 deg: lambda=0 (exact). At 80 deg: r/r_max=0.649 (no saturation). At 89 deg: p_EE->0.
- `dls_lambda_max=5.0` in TDCControllerCfg; `_lf_max` per-env from `set_buoyancy_force()`
- Workspace penalty reward REMOVED (DLS makes it unnecessary)
- Inspired by Yoshikawa manipulability-based adaptive DLS (ALBC 3rd week slides)

## Training Failure Root Causes (pre-DLS, now resolved)
- TDE oscillation in early training -> policy learns "minimize gains" as safe strategy
- K_p=1 gives 0.103 Nm vs passive T_b=3.015 Nm at 45 deg -> no control
- Entropy collapse: noise_std 0.998->0.039 in 456 steps
- Workspace saturation was from TDE terms, not PD (with corrected M_hat=0.14)

## Inertia Values Fixed (2026-02-05)
- Main body: (0.071,0.071,0.031) -> (0.0994,0.0994,0.0372) [URDF R=0.09,L=0.325]
- Buoy: (0.0023,0.0023,0.0034) -> (0.00278,0.00278,0.00336) [URDF R=0.085,H=0.118]

## default_m_hat Fixed (2026-02-05)
- (1.0,1.0) -> (0.14,0.15). True effective inertia: roll=0.0994+0.04=0.1394, pitch=0.0994+0.05=0.1494
- Old default (1.0) was ~7x overestimate, caused workspace saturation from step 1

## Encoder Index Fix (2026-02-05)
- z[:2] was wrong for M_hat; fixed to z[3:5] (roll/pitch in 6-DOF convention)

## Failed TDE Stabilization Approaches (2026-02-06)
All failed because G_loop ~ 119 >> 1 (2nd-order derivative gap between p_EE and coupling).

1. **Decimation sweep** (1,2,5,10): G_loop decreases but PD bandwidth also drops -> error stays 65-69 deg
2. **Rate limiting** (0.005-0.05 m/step): WS_max -> ~1.0 but TDE dominates direction -> 63-69 deg
3. **CC-TDE** (coupling compensation): 3 attempts failed
4. **p_EE command filter (EMA)** alpha sweep (0.05-0.5): alpha=0.05 gave -11% (65->58 deg), all others worse.
   EMA preserves large clamped p_EE in memory, increasing term1 accumulation.

Remaining approaches: Low-pass filtered TDE (UDE), TDE magnitude clamping.

## Archived Insights from Deleted Notes (2026-02-10)

The following insights were extracted from `06_tde_discrete_time_analysis.md` and `07_tdc_analysis_report.md`
before deletion. **CAVEAT**: These analyses were done with Lambda/T_b sign bugs and without DLS IK.
The conclusions about TDE divergence were based on incorrect physics. The conceptual frameworks
may still be useful but numerical results are invalid.

### From 06_tde (Discrete-Time Analysis)
- **G_loop calculation method**: `G_loop = ||M_bm|| * ||J_inv|| / (dt^2 * lf)` -- formula structure valid
  but numerical values assumed wrong Lambda sign
- **Sim-to-real gap framing**: PhysX stiff PD creates unrealistic ddot_Gamma vs real actuator bandwidth
  smoothing. Concept valid regardless of sign bugs.
- **Derivative order mismatch**: Control input (position) vs coupling uncertainty (acceleration) = 2nd order gap.
  This structural observation is sound.

### From 07_report (TDC Analysis Report)
- **UDE tau-spectrum concept**: TDE <-> UDE <-> PD form continuous spectrum via LPF time constant tau.
  tau->0 = TDE, tau->inf = PD. Valid theoretical framework.
- **Sim-to-real actuator smoothing**: Real servos have bandwidth ~5-10 Hz acting as natural LPF on
  joint acceleration. Valid observation independent of sign bugs.
- **Small angle approximation verification**: Confirmed NOT the cause of divergence (PD works fine with
  same approximation). Valid conclusion.
- **History index mismatch verification**: Confirmed no off-by-one error. Valid conclusion.

### Resolution (2026-02-09)
Sign bugs in Lambda/T_b were fixed. DLS IK replaced analytical cosine-law.
TDE saturation removed. nu_dot filter (alpha=0.05) added. PD gains Kp=40, Kd=12.
The TDC controller now works correctly without the divergence these analyses were investigating.
