# Hero Agent Code Review History

## Full Review (2026-03-05, 27 files)
- **Result**: 0 HIGH, 2 MEDIUM (fixed), 4 LOW (no fix needed)
- Fix 1: config.py `3.14/1.5708` -> `math.pi` constants
- Fix 2: base_env.py EMA warm-start pre-reset velocity -> `0.0` (velocity reset to 0 after)

### Verified Correct
- TDC controller math: Lambda signs, TDE formula, anti-windup, EMA filter
- DLS IK: Yoshikawa adaptive damping, convergence
- Proprio history buffer: ring buffer via torch.roll
- Encoder + Adaptation: no double activation, z_hat detach correct, conv dims verified
- Runners: encoder LR cosine decay, checkpoint migration, namespace injection
- Rewards: PBRS Ng 1999 correct, added mass stability clamp, penalty curriculum
- DORAEMON: Beta distribution entropy maximization matches ICLR 2024 paper

## Code Cleanup History

### DELETED Files (Phase 1: Unified/SinglePhase, 2026-02-23)
- `unified_tdc_env.py`, `runners/single_phase_runner.py`, `runners/ppo_aux_mhat.py`

### DELETED Files (Phase 2: Adapt-TDC chain, 2026-02-23)
- `adapt_tdc_env.py` (replaced by adapt_base_env.py)
- ActorCriticEncoderTDC, ActorCriticEncoderTDCAdapt (no longer needed)
- HeroAgentAdaptTDCEnvCfg, RslRlPpoActorCriticEncoderTDCAdaptCfg, HeroAgentAdaptTDCRunnerCfg
- EncoderTDCRewardCfg (dead code)
- NOTE: `encoder_tdc_env.py` NOT deleted, NOT registered (kept as reference)

### DELETED Package (2026-03-05)
- `hero_agent_mpc/` entire package deleted. SAC-MPC experiment concluded.
- Reference: AC-MPC (Romero et al. 2024) adapted for UUV. Details in `sac-mpc-architecture.md`.
