# Time Delay Control (TDC) - Literature Survey

## 1. Origins

TDC was independently proposed by Youcef-Toumi & Ito (1990) and Hsia & Gao (1990).

**Core idea**: Even without knowing the system dynamics, the unknown dynamics can be
estimated using the control input (torque) and state (acceleration) from the immediately
preceding time step. This mechanism is called **Time Delay Estimation (TDE)**.

### Key References

| Year | Authors | Contribution |
|:-----|:--------|:-------------|
| 1990 | Youcef-Toumi & Ito | Original TDC for systems with unknown dynamics |
| 1990 | Hsia & Gao | Independent TDC proposal for robot manipulators |
| 2013 | Jin, Lee, Chang | TDC + Nonlinear Damping + Terminal Sliding Mode |
| 2017 | Cho, Jin, Chang, Lee | Inclusive and Enhanced TDC (IETDC) |
| 2019 | Lee, Chang | Stable Gain Adaptation for TDC |
| 2020 | Chen et al. | Velocity-Free Adaptive TDC |
| 2024 | Taefi et al. | Model-free Adaptive-Robust TDC |

---

## 2. Mathematical Formulation

### Step 1: System Dynamics

General n-DOF robot manipulator dynamics:

```
M(q)q_ddot + C(q, q_dot)q_dot + g(q) + f(q_dot) + d(t) = tau
```

Where:
- `M(q)`: Inertia matrix (n x n, positive definite)
- `C(q, q_dot)q_dot`: Coriolis and centrifugal forces
- `g(q)`: Gravity
- `f(q_dot)`: Friction
- `d(t)`: External disturbances
- `tau`: Control input (torque)

### Step 2: Dynamics Reformulation with M-bar

Introduce an arbitrary **constant diagonal matrix** `M_bar`:

```
M_bar * q_ddot + N(q, q_dot, q_ddot, t) = tau
```

Where `N` is the **lumped uncertainty** (everything unknown):

```
N(q, q_dot, q_ddot, t) = [M(q) - M_bar] * q_ddot + C(q, q_dot)q_dot + g(q) + f(q_dot) + d(t)
```

### Step 3: Time Delay Estimation (TDE)

`N` is unknown, but assuming **N changes negligibly over one sampling period L**:

```
N_hat(t) ~= N(t - L) = tau(t - L) - M_bar * q_ddot(t - L)
```

This is the core of TDE: only past torque `tau(t-L)` and past acceleration `q_ddot(t-L)`
are needed to estimate the current unknown dynamics.

### Step 4: TDC Control Law

```
tau(t) = M_bar * q_ddot_ref(t) + N_hat(t)
       = M_bar * q_ddot_ref(t) + tau(t - L) - M_bar * q_ddot(t - L)
```

### Step 5: Desired Error Dynamics

For tracking error `e = q_d - q`:

```
q_ddot_ref = q_ddot_d + Kd * e_dot + Kp * e
```

Where:
- `Kd`: Derivative gain matrix (diagonal)
- `Kp`: Proportional gain matrix (diagonal)
- `q_ddot_d`: Desired trajectory acceleration

In the ideal case (zero TDE error), the error dynamics become:

```
e_ddot + Kd * e_dot + Kp * e = 0
```

A second-order linear damped system. Gain selection follows standard 2nd-order design:
- Natural frequency: `omega_n = sqrt(Kp)`
- Damping ratio: `zeta = Kd / (2 * omega_n)`

### Complete TDC Block Diagram

```
                   +--------+
  q_d, q_dot_d --> | Desired |
  q_ddot_d ------> | Error   |--> q_ddot_ref --+
  q, q_dot ------> | Dynamics|                 |
                   +--------+                  |
                                               v
                              +----------------------------------+
                              |  tau = M_bar * q_ddot_ref        |
                              |      + tau(t-L)                  |
                              |      - M_bar * q_ddot(t-L)       |
                              +----------------------------------+
                                               |
                                               v
                                           [ Plant ]
                                               |
                                     q, q_dot, q_ddot
```

---

## 3. M-bar Selection

`M_bar` is the single most critical design parameter in TDC.

### Stability Condition

```
|| I - M_bar_inv * M(q) || < 1    (spectral norm)
```

This must hold for all `q` in the workspace.

### Practical Guidelines

| Guideline | Description |
|:----------|:------------|
| Diagonal matrix | `M_bar = diag(m1, m2, ...)` for simplicity |
| Range | Typically 50% ~ 150% of true inertia diagonal elements |
| Too small | TDE error amplified, risk of instability |
| Too large | Sluggish response, poor control performance |
| Nominal inertia | `M_bar ~= M(q_0)` at a representative configuration |

### Effect on Closed-Loop

- `M_bar` acts as a **bandwidth limiter**: larger `M_bar` reduces sensitivity but
  also reduces responsiveness.
- The ratio `M_bar_inv * M(q)` determines how well TDE cancels the true dynamics.
  When `M_bar = M(q)`, cancellation is perfect (but then you already know the model).

---

## 4. TDE Error Analysis

In practice `N(t) != N(t-L)`, so TDE error `epsilon` exists:

```
epsilon(t) = N(t) - N(t-L)
```

### Error Characteristics

| Factor | Effect on TDE Error |
|:-------|:--------------------|
| Smaller L (sampling period) | Smaller error |
| Faster-changing dynamics | Larger error |
| Higher accelerations | Larger error (N depends on q_ddot) |

### Stability Under TDE Error

The closed-loop error dynamics with TDE error:

```
e_ddot + Kd * e_dot + Kp * e = M_bar_inv * epsilon(t)
```

**Ultimate boundedness** (Lyapunov-based):
- The tracking error is bounded, not asymptotically zero.
- Bound proportional to `|| M_bar_inv * epsilon ||`.

### Mitigation Strategies

1. **Minimize L**: Use the fastest feasible sampling rate.
2. **Sliding mode compensation**: Add a sliding mode term to reject TDE error.
3. **Adaptive M_bar**: Adjust M_bar online to reduce `|| I - M_bar_inv * M(q) ||`.
4. **Disturbance observer**: Add a secondary observer for TDE error estimation.

---

## 5. Variants

### 5.1 Standard TDC (Youcef-Toumi, 1990)

```
tau = M_bar * (q_ddot_d + Kd * e_dot + Kp * e) + tau(t-L) - M_bar * q_ddot(t-L)
```

Simple, effective, but TDE error is uncompensated.

### 5.2 TDC + Sliding Mode

Adds a switching term to handle TDE error:

```
tau = M_bar * q_ddot_ref + N_hat + M_bar * K_s * sign(s)
```

Where `s` is a sliding surface (e.g., `s = e_dot + lambda * e`).
Reduces tracking error bound but may introduce chattering.

### 5.3 Enhanced TDC (IETDC, Cho et al. 2017)

Three components:
1. TDE (standard)
2. Nonlinear desired error dynamics (DED)
3. TDE error correction via nonlinear sliding surface

Improves robustness without chattering.

### 5.4 Adaptive TDC

- **Adaptive M_bar**: Online tuning of `M_bar` using Nussbaum function or gradient.
- **Adaptive gains**: `Kp`, `Kd` adjusted based on tracking performance.

### 5.5 TDC + Disturbance Observer

Uses a secondary observer to estimate and compensate the TDE error `epsilon(t)`,
providing tighter tracking bounds.

---

## 6. Underwater Vehicle Applications

TDC is particularly well-suited for underwater robots because:

1. **Complex hydrodynamics**: Added mass, nonlinear damping, Coriolis forces are
   difficult to model precisely. TDC bypasses this requirement.
2. **Environmental disturbances**: Ocean currents and wave forces are naturally
   handled as part of the lumped uncertainty `N`.
3. **Parameter variation**: Buoyancy and added mass change with depth, payload,
   and configuration. TDC adapts implicitly via TDE.

### Relevant Work

- **Nonlinear robust control of UVMS based on TDE** (IEEE ICRA 2017):
  Applied to underwater vehicle-manipulator systems with coupled dynamics.
- **Robust trajectory control of underwater vehicles using TDC** (Ocean Engineering 2006):
  Demonstrated TDC for 6-DOF trajectory tracking of AUVs.
- **Chattering-suppression SMC with TDE for underwater manipulator** (JMSE 2023):
  Combined TDE with continuous sliding mode for smooth control.

---

## 7. Practical Design Guidelines

### Parameter Selection Summary

| Parameter | Guideline | Typical Range |
|:----------|:----------|:--------------|
| `L` (time delay) | = control loop period, as small as possible | 1 ~ 10 ms |
| `M_bar` | Diagonal, ~50-150% of true inertia | System-dependent |
| `Kp` | `omega_n^2` for desired natural frequency | 10 ~ 1000 |
| `Kd` | `2 * zeta * omega_n` for desired damping | 1 ~ 100 |
| `zeta` | Damping ratio | 0.7 ~ 1.0 (critically damped) |

### Acceleration Measurement

The main practical challenge: TDE requires `q_ddot(t-L)`.

| Method | Pros | Cons |
|:-------|:-----|:-----|
| Numerical differentiation | Simple | Noisy, amplifies sensor noise |
| Low-pass filtered derivative | Reduces noise | Introduces phase lag |
| Observer (e.g., Kalman) | Optimal estimation | More complex |
| IMU (for attitude systems) | Direct measurement | Sensor cost, drift |

### Digital Implementation

```
# At each control step k (period = L):
q_ddot_prev = (q_dot[k-1] - q_dot[k-2]) / L     # or from observer
tau_prev     = tau[k-1]                             # stored from last step

N_hat = tau_prev - M_bar * q_ddot_prev              # TDE

e     = q_d[k] - q[k]
e_dot = q_dot_d[k] - q_dot[k]
q_ddot_ref = q_ddot_d[k] + Kd * e_dot + Kp * e     # desired error dynamics

tau[k] = M_bar * q_ddot_ref + N_hat                 # TDC control law
```

---

## 8. Key Takeaways for Implementation

1. **Model-free**: No need for dynamics identification. Only `M_bar` (rough inertia
   estimate) is required.
2. **Three design parameters**: `M_bar`, `Kp/Kd`, and `L`.
3. **Acceleration estimation** is the main practical challenge. For simulation,
   this is trivially available from the physics engine.
4. **Standard TDC is sufficient** as a starting point. Sliding mode or adaptive
   extensions can be added later if TDE error causes problems.
5. **Stability is guaranteed** as long as `|| I - M_bar_inv * M(q) || < 1` and
   `L` is sufficiently small.

---

## References

- Youcef-Toumi, K., & Ito, O. (1990). A Time Delay Controller for Systems with Unknown Dynamics. ASME J. Dynamic Systems, Measurement, and Control.
- Hsia, T.C., & Gao, L.S. (1990). Robot Manipulator Control Using Decentralized Linear Controller and Nonlinear Disturbance Observer. IEEE ICRA.
- Cho, G.R., Jin, M., Chang, P.H., & Lee, J. (2017). Robust Control of Robot Manipulators Using Inclusive and Enhanced Time Delay Control. IEEE/ASME Transactions on Mechatronics.
- Jin, M., Lee, J., & Chang, P.H. (2013). Stability Guaranteed Time-Delay Control Using Nonlinear Damping and Terminal Sliding Mode. IEEE Transactions on Industrial Electronics.
- Lee, J., & Chang, P.H. (2019). Stable Gain Adaptation for Time-Delay Control. IFAC-PapersOnLine.
- Chen, Z., et al. (2020). Velocity-Free Adaptive Time Delay Control of Robotic System. Mathematical Problems in Engineering.
