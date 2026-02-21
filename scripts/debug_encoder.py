"""Debug script: Load encoder model and analyze z output behavior.

Loads model_0.pt (initial) and a later checkpoint, then:
1. Checks encoder weight statistics
2. Feeds random privileged observations through encoder
3. Checks raw encoder output (before softplus) and final z
4. Checks gradient flow
"""

import torch
import torch.nn.functional as F

# Load model checkpoints
baseline_path = "/workspace/isaaclab/logs/rsl_rl/hero_agent_albc_encoder/baseline_encoder-base/model_0.pt"
current_path = "/workspace/isaaclab/logs/rsl_rl/hero_agent_albc_encoder/2026-02-21_01-54-22_encoder-base/model_0.pt"
current_late_path = "/workspace/isaaclab/logs/rsl_rl/hero_agent_albc_encoder/2026-02-21_01-54-22_encoder-base/model_1100.pt"

baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
current = torch.load(current_path, map_location="cpu", weights_only=False)
current_late = torch.load(current_late_path, map_location="cpu", weights_only=False)

# Find encoder keys
print("=" * 60)
print("ENCODER WEIGHT COMPARISON (model_0)")
print("=" * 60)

model_state = baseline["model_state_dict"]
encoder_keys = sorted([k for k in model_state.keys() if "encoder" in k])
print(f"\nEncoder keys: {encoder_keys}")

for key in encoder_keys:
    b_val = baseline["model_state_dict"][key]
    c_val = current["model_state_dict"][key]
    diff = (b_val - c_val).abs().max().item()
    print(f"\n{key}:")
    print(f"  Baseline: mean={b_val.mean().item():.6f}, std={b_val.std().item():.6f}, "
          f"min={b_val.min().item():.6f}, max={b_val.max().item():.6f}")
    print(f"  Current:  mean={c_val.mean().item():.6f}, std={c_val.std().item():.6f}, "
          f"min={c_val.min().item():.6f}, max={c_val.max().item():.6f}")
    print(f"  Max diff: {diff:.8f}")

# Check if initial weights are IDENTICAL
all_same = True
for key in encoder_keys:
    if not torch.equal(baseline["model_state_dict"][key], current["model_state_dict"][key]):
        all_same = False
        break
print(f"\nEncoder initial weights identical? {all_same}")

# Check ALL keys for differences
print("\n" + "=" * 60)
print("ALL WEIGHT DIFFERENCES (model_0)")
print("=" * 60)
any_diff = False
for key in sorted(model_state.keys()):
    b_val = baseline["model_state_dict"][key]
    c_val = current["model_state_dict"][key]
    if not torch.equal(b_val, c_val):
        diff = (b_val - c_val).abs().max().item()
        print(f"  DIFF: {key}: max_diff={diff:.8f}")
        any_diff = True
if not any_diff:
    print("  ALL weights are IDENTICAL at model_0!")

# Now manually run the encoder forward pass
print("\n" + "=" * 60)
print("ENCODER FORWARD PASS TEST")
print("=" * 60)

# Reconstruct encoder MLP from weights
# Encoder: 26D -> [256, 128, 64] -> 13D
# Layer structure in rsl_rl MLP: Linear(in, hidden[0]), activation, Linear(hidden[0], hidden[1]), ...

# Create test input: representative privileged observations
# main_hydro(7): volume~0.008, CoG~(-0.05,0,0), CoB~(0,0,0), etc
# buoy_hydro(7): volume~0.003, CoG/CoB~0
# main_dynamics(4): Ixx~0.1, Iyy~0.1, Izz~0.04, m_A~0.6
# buoy_dynamics(4): Ixx~0.003, Iyy~0.003, Izz~0.003, m_A~0.15
# payload(4): mass~0.5, cog~(0,0,-0.05)

torch.manual_seed(42)
test_input = torch.randn(16, 26) * 0.1  # Small random values like typical privileged obs

# Also try with realistic ranges
realistic_input = torch.tensor([
    [0.008, 0.0, 0.0, -0.05, 0.0, 0.0, 0.0,  # main hydro
     0.003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    # buoy hydro
     0.1, 0.1, 0.04, 0.6,                       # main dynamics
     0.003, 0.003, 0.003, 0.15,                  # buoy dynamics
     0.5, 0.0, 0.0, -0.05]                       # payload
]).expand(16, -1)

for name, inp in [("random", test_input), ("realistic", realistic_input)]:
    print(f"\n--- Input: {name} ---")

    for model_name, model_data in [("baseline_0", baseline), ("current_0", current), ("current_1100", current_late)]:
        sd = model_data["model_state_dict"]

        # Find encoder layer weights (MLP structure)
        # rsl_rl MLP stores as architecture.0.weight, architecture.2.weight, etc. (with ReLU at 1, 3, etc.)
        enc_weights = []
        enc_biases = []
        i = 0
        while True:
            wkey = f"encoder.architecture.{i}.weight"
            bkey = f"encoder.architecture.{i}.bias"
            if wkey not in sd:
                break
            enc_weights.append(sd[wkey])
            enc_biases.append(sd[bkey])
            i += 2  # Skip activation layers

        # Manual forward pass
        x = inp.clone()
        for j, (w, b) in enumerate(zip(enc_weights, enc_biases)):
            x = F.linear(x, w, b)
            if j < len(enc_weights) - 1:  # Apply ReLU for all but last layer
                x = F.relu(x)

        raw_output = x
        z = F.softplus(raw_output) + 0.01  # z_min = 0.01

        print(f"  {model_name}: raw_out mean={raw_output.mean().item():.4f}, "
              f"std={raw_output.std().item():.4f}, "
              f"min={raw_output.min().item():.4f}, max={raw_output.max().item():.4f}")
        print(f"  {model_name}: z mean={z.mean().item():.4f}, "
              f"std={z.std().item():.4f}, "
              f"min={z.min().item():.4f}, max={z.max().item():.4f}")

# Check actor weights for differences
print("\n" + "=" * 60)
print("ACTOR/CRITIC WEIGHT COMPARISON (model_0 vs model_1100)")
print("=" * 60)

for key in sorted(current["model_state_dict"].keys()):
    c0_val = current["model_state_dict"][key]
    c1100_val = current_late["model_state_dict"][key]
    diff = (c0_val - c1100_val).abs().max().item()
    if diff > 1e-6:
        print(f"  CHANGED: {key}: max_diff={diff:.6f}")

# Check std parameter
print("\n" + "=" * 60)
print("STD PARAMETER")
print("=" * 60)
for name, model_data in [("baseline_0", baseline), ("current_0", current), ("current_1100", current_late)]:
    sd = model_data["model_state_dict"]
    if "std" in sd:
        print(f"  {name}: std = {sd['std']}")
    if "log_std" in sd:
        print(f"  {name}: log_std = {sd['log_std']}")

# Check optimizer state for encoder
print("\n" + "=" * 60)
print("OPTIMIZER STATE CHECK (do encoder params get updates?)")
print("=" * 60)
if "optimizer_state_dict" in current_late:
    opt = current_late["optimizer_state_dict"]
    param_groups = opt.get("param_groups", [])
    print(f"  Param groups: {len(param_groups)}")
    for i, pg in enumerate(param_groups):
        print(f"  Group {i}: lr={pg.get('lr')}, #params={len(pg.get('params', []))}")

    # Check if encoder parameters have accumulated gradient moments
    state = opt.get("state", {})
    # Find which param indices correspond to encoder
    all_keys = list(current_late["model_state_dict"].keys())
    encoder_param_indices = [j for j, k in enumerate(all_keys) if "encoder" in k]
    print(f"  Encoder param indices (in state_dict order): {encoder_param_indices}")

    # Check optimizer state for first encoder param
    if encoder_param_indices:
        idx = encoder_param_indices[0]
        if idx in state:
            s = state[idx]
            print(f"  Encoder param[{idx}] optimizer state keys: {list(s.keys())}")
            if "exp_avg" in s:
                print(f"    exp_avg: mean={s['exp_avg'].abs().mean().item():.8f}, max={s['exp_avg'].abs().max().item():.8f}")
            if "exp_avg_sq" in s:
                print(f"    exp_avg_sq: mean={s['exp_avg_sq'].mean().item():.8f}, max={s['exp_avg_sq'].max().item():.8f}")
            if "step" in s:
                print(f"    step: {s['step']}")
        else:
            print(f"  WARNING: Encoder param[{idx}] NOT in optimizer state!")
            print(f"  Available state keys: {sorted(state.keys())[:20]}...")
