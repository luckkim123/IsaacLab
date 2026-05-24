# Repo 3-Split Design: isaaclab / marinelab / constrained-albc

**Date:** 2026-05-25
**Branch:** feat/encoder-tdc-integration
**Status:** Design approved, pending implementation plan
**Context:** Task #4 of the 2026-05-24 workspace cleanup. Split the single isaaclab working
tree into three purpose-scoped repositories with an enterprise-standard overlay architecture.

---

## 1. Goal & Motivation

The current `/workspace/isaaclab` tree mixes three concerns in one fork:

- upstream IsaacLab framework code (v2.3.0-based),
- a public underwater-environment layer (bluerov + UUV assets + shared marine physics),
- private research code (constrained ALBC + TDC + student distillation).

This entangles upstream tracking, complicates junior onboarding, and blurs the public/private
boundary. The split produces three independently installable, documented repositories that
layer cleanly:

- **isaaclab** -- clean upstream fork, Docker base layer.
- **marinelab** -- public overlay extension (deploy to juniors).
- **constrained-albc** -- private research overlay (`visibility=private`).

Naming follows the HORA/RMA convention (method/domain short-name as repo name; privacy handled
by GitHub repo visibility, not by name suffix).

---

## 2. Architecture

Three-layer overlay, bottom to top. Each upper layer `pip install`s and imports the layer below;
there are no upward dependencies.

```
isaaclab            (clean fork: isaac-sim/IsaacLab v2.3.0; Docker base; zero of our code)
  ^  pip install / import isaaclab
marinelab           (external ext: bluerov env + uuv assets[Git LFS] + shared marine physics)
  ^  pip install / import marinelab
constrained-albc    (external ext: constrained_full_albc + _tdc + student + analysis)
```

### Verified dependency facts (grep-checked, not assumed)

- **Shared marine core** = `isaaclab_tasks/models/{hydrodynamics,thruster}.py` +
  `assets/robots/uuv/uuv_cfg.py`. Imported by BOTH bluerov (public) AND constrained_full_albc
  (private). -> assigned to marinelab (lower layer owns the shared physics).
- **bluerov <-> albc cross-imports = 0** (checked both directions). Clean separation.
- **albc -> shared core is one-directional.** Fixes layer order: marinelab below, albc above.

### Overlay method: external-extension self-registration

marinelab and constrained-albc each register their Gym environments via their own
`pyproject.toml` entry-points / `extension.toml`, instead of editing isaaclab's core
`__init__.py`. This is IsaacLab's documented external-project pattern and keeps the isaaclab
fork a zero-touch clean fork (upstream rebase has no conflicts).

---

## 3. Upstream Core-Touch Resolution

Investigation found exactly **8 upstream-core files touched by our commits and still present at
HEAD**. Attribution was done by first-commit author (the v2.3.0-tag presence test is unreliable
because we forked before v2.3.0 and later absorbed post-v2.3.0 upstream commits).

| # | File | Verified finding | Resolution |
|:--|:--|:--|:--|
| 1 | `isaaclab/utils/volume.py` | Ours (luckkim123, 2026-02-05); buoyancy volume calc | Move to `marinelab/utils/volume.py` |
| 2 | `isaaclab/utils/__init__.py` | Only `from .volume import *` is ours. `.logger` (Mayank), `.mesh` (renezurbruegg) are upstream | Removing the volume line fully reverts our footprint |
| 3 | `isaaclab/sim/converters/urdf_converter.py` | NOT ours -- all substantive edits are Mayank/Kelly (version pins, ruff isort). No UUV refs. Our "backup" commit changed nothing here | Drop our delta; clean fork keeps upstream version |
| 4 | `isaaclab_rl/.../rsl_rl/rl_cfg.py` | `state_dependent_std` (in `RslRlPpoActorCriticCfg`) + `weight_decay` (in `RslRlPpoAlgorithmCfg`); both upstream `@configclass` dataclasses | Subclass in constrained-albc. **VERIFY at impl time** (see Open Question) |
| 5 | `isaaclab_assets/robots/__init__.py` | `from .uuv import *` registration | Revert; marinelab self-registers |
| 6 | `isaaclab_tasks/direct/__init__.py` | `from . import bluerov` registration | Revert; marinelab self-registers |
| 7 | `isaaclab_tasks/__init__.py` | `_BLACKLIST_PKGS += "models"` | Revert; `models/` leaves for marinelab |
| 8 | `isaaclab_tasks/test/test_tdc_controller.py` | Our test | Move to `constrained-albc/tests/` |

**Net effect:** isaaclab reverts to a clean fork; marinelab and constrained-albc become
near-pure overlays. The only genuine implementation-time risk is item #4.

### Open Question (impl-time verification)

Does the **rsl_rl library runtime** read `state_dependent_std` / `weight_decay` by attribute
(subclassing works) or against a fixed schema (subclassing insufficient)? Must be verified
before relying on the subclass approach in constrained-albc. Fallback if fixed-schema: keep a
documented thin patch to `isaaclab_rl` or vendor the two fields.

---

## 4. Deployment & Docker Layering

Junior-facing deploy story: build isaaclab as a Docker base, then overlay marinelab.

```
isaaclab fork Docker image           (NVIDIA Isaac Sim 5.1 + isaaclab v2.3.0; built from our
                                       clean fork's docker/ config -- source-build, the standard
  | FROM                              IsaacLab path)
  v
marinelab Dockerfile                 (in marinelab repo; junior deploy entry point)
  FROM <isaaclab-image>
  RUN pip install -e marinelab
  RUN git lfs pull                   (uuv meshes, 140MB)
  |  (researcher-only, optional)
  v
constrained-albc                     pip install -e .  (on top of marinelab; private)
```

**Docker base decision:** build from our isaaclab clean fork (option a), matching IsaacLab's
own source-build-first convention (reproducibility + control; identical to the dev container).

**marinelab repo ships for juniors:** `Dockerfile`, `docker-compose.yaml` (GPU/X11/volume
mounts, reusing isaaclab's `docker/` patterns), README install steps, and Git LFS setup notes.

**Deploy verification:** on a fresh machine -- build isaaclab Docker -> `pip install` marinelab
-> confirm `Isaac-Bluerov-*` Gym registration -> run `random_agent` once successfully.

---

## 5. Repository Internal Structure & Documentation

### Common documentation standard (all three repos)

```
README.md          quickstart + install + demo image
docs/
  installation.md  Docker + local
  architecture.md  structure + dependency diagram
CONTRIBUTING.md    (esp. important for marinelab juniors)
CHANGELOG.md       Keep-a-Changelog format
LICENSE            BSD-3 for isaaclab/marinelab (upstream-inherited); private for constrained-albc
pyproject.toml     package metadata + entry-point (Gym self-register)
.gitignore
```

### marinelab layout (external extension)

```
marinelab/
  marinelab/
    __init__.py            extension registration entry point
    assets/                uuv_cfg + bluerov + hero_agent (Git LFS-tracked meshes)
    physics/               hydrodynamics.py, thruster.py   (was isaaclab_tasks/models/)
    utils/                 volume.py                        (was isaaclab/utils/volume.py)
    tasks/
      bluerov/             bluerov env + tasks + mdp
  docker/                  Dockerfile (FROM isaaclab) + compose
  docs/  README.md  pyproject.toml  ...
```

### constrained-albc layout (external extension)

```
constrained-albc/
  constrained_albc/
    envs/                  constrained_full_albc + constrained_full_albc_tdc
    algorithms/            constraint_trpo, IPO
    encoder/  runners/  mdp/
    student/               TCN/GRU distillation
    analysis/              eval_dr, train-analyze tools (was scripts/analysis/)
  scripts/                 launch_*.sh (incl. former scripts/student/), train/eval entry points
  tests/                   test_tdc_controller.py
  docs/  README.md  pyproject.toml  ...
```

### Key restructurings (in-tree -> repo)

- `isaaclab_tasks/models/` -> `marinelab/physics/` (clearer name)
- `scripts/analysis/` -> `constrained-albc/constrained_albc/analysis/` (into the package)
- `scripts/student/` -> `constrained-albc/scripts/` (the 4 student `.sh` were grouped under
  `scripts/student/` in commit d0c54a74 as a pre-split staging step)

---

## 6. Migration Procedure & Verification

Migrate bottom layer first. The current `/workspace/isaaclab` remains the source of truth and is
NOT deleted until all three repos are split and verified.

### Phase 0 -- isaaclab clean-fork restoration

- Revert/relocate the 8 core-touches per Section 3.
- Verify: `import isaaclab` OK; no UUV/albc code remains; diff against upstream shows none of our
  additions.

### Phase 1 -- marinelab repo

- Move `models/` + uuv assets + bluerov into the marinelab layout.
- Add `extension.toml` + `pyproject.toml` entry-point self-registration.
- Configure Git LFS for `*.dae`, `*.obj`, `*.usd`.
- Add Docker (FROM isaaclab) + docs.
- Verify: on isaaclab Docker, `pip install` marinelab -> `Isaac-Bluerov-*` registers ->
  `random_agent` runs.

### Phase 2 -- constrained-albc repo

- Move constrained_full_albc + _tdc + student + analysis into the layout.
- Declare marinelab dependency (physics/assets imports).
- Apply rl_cfg subclass (verify rsl_rl runtime behavior -- Section 3 #4).
- Verify: on marinelab, install constrained-albc -> `Isaac-FullDOF-TRPO-v0` registers ->
  `eval_dr.py static` runs.

### git history preservation (per-repo, by character)

- **constrained-albc**: preserve history via `git filter-repo` on the relevant paths. Research
  blame value is high (tracing why a reward term or constraint evolved).
- **marinelab**: clean start (current files + one initial commit). Junior-facing; a clean slate
  is preferable to inheriting research trial-and-error history.
- Either way, full history remains in the isaaclab fork for archival reference.

### Safety constraints (memory rules)

- Do NOT delete current isaaclab until split is complete and verified (source of truth).
- Move `model_*.pt` and logs by numeric sort only (model-trim-disaster rule); dry-run before any
  deletion.
- No `git push` this session (user defers push).

---

## 7. Out of Scope

- Actual upstream PRs for #3 urdf_converter (mergeable but separate effort).
- CI/CD pipeline setup for the new repos.
- Migrating `logs/` and `wandb/` artifacts (handled separately per cleanup worktree-lifecycle rule).
