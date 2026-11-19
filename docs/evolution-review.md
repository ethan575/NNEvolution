# Review of the evolution / fitness code

Scope: `NeuralNet/FitnessEval.cs`, `NeuralNet/NNTrainer.cs`, `Ragdoll/RagdollBody.cs`,
`Scenes/SimulationScene.cs`, `Scenes/HeadlessSimulation.cs`, `Ragdoll/Joint.cs`.

The short version: **nothing is crashing, and the population is not misbehaving — it is
doing exactly what the objective pays for.** The objective pays for net metres of
displacement toward a teleporting target, and it *punishes* getting faster unless you
also happen to be a near-perfect turner. That produces a wide, comfortable basin at
"drift in one fixed direction" and a ridge everywhere else. Your "bottom-left is
efficient, the right is a struggle" is the shape of that fitness landscape.

Findings are ordered by how much they matter for that symptom.

---

## 1. CRITICAL — the fitness function has no term for *steering*, and the term that
## looks like it does is mis-specified

`Classes/NeuralNet/FitnessEval.cs:47-78`

```csharp
foreach (float progressVal in ragdoll.PhaseProgressMeters) { ... }
float minPosBonus = MathF.Max(0f, minPhase); // smallest positive progress give bonus
float arrivalBonus = ragdoll.PhasesReachedCount * 2f;
return positiveSum + (0.5f * negativeSum) + (2.0f * minPosBonus) + arrivalBonus;
```

Three problems:

**(a) `minPhase` is not a direction measure.** The comment says it "favours moving in
all 4 directions", but `PhaseProgressMeters` holds *net metres of progress during each
phase* — a scalar. The target directions are never consulted (no `JVector` is used
anywhere in this method). So `minPhase` is "your worst phase", and `minPosBonus` is
really **"bonus for never having a bad phase"**. A creature that walks in a straight
line gets every phase positive, so it collects `2.0 × min(progress)` *on top of* the
full positive sum. A creature that attempts a turn and fails on one phase loses three
times: the phase enters `negativeSum` at half weight, it drives `minPhase <= 0` and
zeroes the 2× bonus, and it can cost an arrival.

**(b) The reward is in metres and unbounded, while everything else is O(1).** Phase
targets sit ~39 m away and a phase lasts 6.67 s, so raw speed dominates the score.
The accuracy signal is a flat `+2` per arrival. There is no shaping that rewards
"close to the target at the end of the phase" beyond that.

**(c) Monotonicity argument — this is the trap.** Because the score is
`Σ progress − 0.5·|negative|`, it is only monotone in speed *if you are a perfect
tracker*. For an imperfect turner, moving faster is **strictly worse**. Numbers from a
Monte-Carlo replication of `initializeGenTargetSchedule` + `EvalMultiPhaseTargetFollowing`
(4000 generations per cell, 40 substeps per phase so min-distance is realistic):

Policy = walk a fixed world direction at speed `v`, no turning:

| v (m/s) | 0.5 | 1.0 | 2.0 | 3.0 | 6.0 |
|---|---|---|---|---|---|
| mean fitness | +1.10 | +1.20 | **−2.13** | **−10.18** | **−38.86** |

Perfect target tracker at speed `v`:

| v (m/s) | 0.5 | 1.0 | 2.0 | 3.0 | 6.0 |
|---|---|---|---|---|---|
| mean fitness | +16.7 | +33.3 | +66.7 | +100.0 | +194.3 |

So: the straight walker peaks at ~1 m/s and **gets worse the faster it goes**, while the
only way to earn more is to become a tracker. The population therefore parks at
"slow-ish drift in one direction" — cheap, positive, and local. Any mutation that raises
speed *and* not turning is selected against.

Add turning authority and the valley is explicit (fitness of a perfect-knowledge
steerer with max turn rate `w`, 600 gens/cell):

| w (deg/s) | v=1.0 | v=1.5 | v=2.5 | v=4.0 | v=6.0 |
|---|---|---|---|---|---|
| 0 | 1.2 | 0.1 | −5.9 | −19.6 | −38.6 |
| 10 | 2.7 | 3.3 | 2.1 | −3.9 | −11.8 |
| 20 | 14.0 | 19.8 | 28.7 | 39.7 | 56.5 |
| 40 | 23.9 | 35.6 | 57.8 | 89.0 | 125.4 |
| 90 | 29.4 | 44.0 | 72.5 | 115.2 | 170.4 |

Below roughly **10–20 °/s of effective turn authority the steering signal is
indistinguishable from noise**. A hinge chain with ±45° limits, 400 N·m motors on ~1 kg
bones and 10 rad/s cap does not obviously clear that bar, and it must clear it *while
keeping its speed*. That is a ridge, not a slope — a textbook premature-convergence
geometry. (A GA seeded at a 10 °/s straight-line policy with this same reward does
converge to max-speed + max-turn in the abstract limit, so the *optimum* is a turner;
the problem is the basin around the straight walker is enormous.)

### Fix

Make each phase a **bounded, relative** score, and stop paying for "worst phase isn't
negative":

```csharp
// FitnessEval.cs
public static float EvalMultiPhaseTargetFollowing(RagdollBody ragdoll)
{
    if (ragdoll.PhaseProgressMeters.Count == 0)
        return 20f * EvalProgressToTarget(ragdoll.initialPos, ragdoll.COMPosition, ragdoll.TargetPosition);

    float total = 0f;
    for (int i = 0; i < ragdoll.PhaseProgressMeters.Count; i++)
    {
        float start = MathF.Max(ragdoll.PhaseStartDistances[i], 1e-3f);
        // relative progress: -1 (walked directly away) .. +1 (arrived)
        total += Math.Clamp(ragdoll.PhaseProgressMeters[i] / start, -1f, 1f);
    }
    // arrival must dominate everything else
    return total + 3f * ragdoll.PhasesReachedCount;
}
```

with `PhaseStartDistances` (a `List<float>` filled in `StartTargetPhase`, alongside
`PhaseProgressMeters`) recorded in `RagdollBody.FinalizeCurrentPhase`
(`Classes/Ragdoll/RagdollBody.cs:176-189`). Now a straight-line walker earns at most
`3 × (39−39)/39 ≈ 0` per phase regardless of how fast it is, and `+1` is reserved for
actually arriving. Speed no longer buys anything on its own.

---

## 2. CRITICAL — the task as configured is close to physically impossible, and the
## clamp is not directionally neutral

`Classes/Scenes/SimulationScene.cs:508-535`

- 3 targets, `120°` apart, radius **32–46** from spawn → consecutive targets are
  `39·√3 ≈ 67.5 m` apart.
- Phase duration is `20/3 ≈ 6.67 s`.
- So to *arrive* (10 m radius) you need `(67.5 − 10)/6.67 ≈ 8.6 m/s` average **and** a
  120° turn **and** a standing start. Just reaching a target at 39 m from rest needs
  `5.85 m/s`.

Your walkers top out around 1–2 m/s. `PhasesReachedCount` will therefore be ~0 for
essentially the whole run, which means the only live term is phase progress — the
treadmill in §1. This is why "efficient" is the only thing that ever improves.

**The clamp is also not radial:**

```csharp
float x = Math.Clamp(centre.X + MathF.Cos(angles[i]) * radius, -52f, 52f);
float z = Math.Clamp(centre.Z + MathF.Sin(angles[i]) * radius, -52f, 52f);
```

Clamping X and Z independently moves targets *toward* whichever axis is being clipped.
If the creature spawns off-centre (e.g. COM ≈ (20, 10); the spawn point comes straight
from the `.creature` file), a 120° away target at (60, 10) becomes (52, 10) — **32 m**
away instead of 46 — while the others stay at 40 m. That is a *stochastic reward for
facing a particular world direction*: the straight-line walker does better when its
drift happens to point at a clipped corner. That is a plausible mechanism for locking
in one specific direction ("bottom left") rather than just one arbitrary direction.

Suggested fix:

- Shrink the ring: `radius 12..18` (still a real walk, still a real turn) — or raise
  `GenerationDuration` so 3 phases are ~15 s each.
- Clamp radially, so all three targets keep equal distance:
  ```csharp
  JVector off = new JVector(MathF.Cos(a), 0, MathF.Sin(a)) * radius;
  float len = off.Length();
  float maxLen = 52f - MathF.Max(MathF.Abs(centre.X), MathF.Abs(centre.Z));
  if (len > maxLen) off *= maxLen / len;
  phaseTargets[i] = new JVector(centre.X + off.X, TargetFloorY, centre.Z + off.Z);
  ```
- Consider **per-phase reset** (see §3) so a phase is "from here, get to that" rather
  than a continuous 20 s chain. That removes the treadmill completely and equalises all
  directions.

---

## 3. HIGH — the target cube is a dynamic body that only exists in world 0, so one
## creature can move the target for the whole population, and the elite is scored in a
## different world from everyone else

`Classes/Scenes/SimulationScene.cs:405-413`, `:100-121`, `:182`

- `AllSimWorlds[0]` is constructed over `ForegroundWorld`
  (`new HeadlessSimulationScene(ForegroundWorld, ForegroundBody)`), and
  `ForegroundWorld` also contains the floor **and** `TargetCube`. Worlds 1..49 get a
  fresh `World` with only a floor. So the foreground creature can physically shove,
  block or topple the target while nobody else can. The elite is measured under
  different physics from its competitors.
- Every world steers using `liveTargetPos = TargetCube.Rbody.Position`
  (`:111`), so **if the foreground creature pushes the cube, the target moves for all
  50 creatures**.
- Scoring uses a *different* position: `FinalizeCurrentPhase` compares against
  `phaseTargetPos` (the intended point), while `BrainNNFF`/`GetBrainInputMatrix` steer
  to `TargetPosition` (the live point). The cube has `setObeyGravity(true)` and is only
  teleported on phase change, so drift is real. Scoring and steering disagree.

Fix: keep `TargetCube` out of collision with creatures (put it in a separate
`World`, or give it `MotionType.Static` during generations and make it kinematic), and
score the phase against the **same** value the network was given. Simplest: make the
target kinematic (`MotionType.Kinematic` + teleport per phase) and drop the foreground
cube from the foreground world's collision set.

---

## 4. HIGH — the self-collision filter turns self-collision off entirely, not just for
## adjacent bones

`Classes/Scenes/HeadlessSimulation.cs:20-32`

```csharp
if (ragdollBodies.Contains(shapeA.RigidBody) && ragdollBodies.Contains(shapeB.RigidBody))
    return false; // false = skip narrowphase (per Jitter2 IBroadPhaseFilter docs)
```

Every bone pair of the same creature is skipped, so **all limbs interpenetrate freely**.
The intent was clearly "adjacent bones don't fight each other", but as written a limb can
pass straight through the torso. That materially changes which gaits are cheap: a
"paddle through the body" gait is free, and compact vs. sprawled postures are not
differentiated. It is also a per-creature advantage that evolution will happily exploit.

Fix: filter only pairs that share a joint.

```csharp
// build this in the RagdollBody / filter ctor
private readonly HashSet<(RigidBody, RigidBody)> adjacent = ...; // from ragdoll.Joints

public bool Filter(IDynamicTreeProxy a, IDynamicTreeProxy b)
{
    if (a is RigidBodyShape sa && b is RigidBodyShape sb)
        if (adjacent.Contains((sa.RigidBody, sb.RigidBody)) ||
            adjacent.Contains((sb.RigidBody, sa.RigidBody)))
            return false;
    return true;
}
```

---

## 5. MEDIUM — the network's reference frame is an arbitrary bone

`Classes/Ragdoll/RagdollBody.cs:204`

```csharp
JQuaternion refOrientation = Bones.Count > 0 ? Bones[0].RB.Orientation : JQuaternion.Identity;
JQuaternion invRefOrientation = JQuaternion.Conjugate(refOrientation);
```

`Bones[0]` is whatever was first in the `.creature` file. Three consequences:

- "Direction to target in local frame" is expressed in a frame whose axes have no fixed
  relationship to the creature's forward/up. A network that learns "input +X ⇒ yaw
  left" gets the sign flipped whenever that particular bone rolls.
- `Bones[0]` is *also* used for `localUp`, so tilt feedback is equally arbitrary.
- Bone order is not guaranteed stable across saves, so a `.nn` trained against one file
  ordering can be applied to another with the meaning of 6 of its 12 proprioceptive
  inputs changed.

Fix: define the reference frame explicitly — e.g. the *pelvis/root* bone if the format
records it, or better, build an orthonormal body frame from the COM-to-root vector plus
world-up, and express target/velocity/up in that. Also derive it once per bone and
store it on the body instead of hardcoding index 0.

Related, same method:

- `dist / 30f` (line 242) saturates at 30, but targets are 32–46 m away → this input is
  **pinned at 1.0 for the whole generation** and carries no information.
- `worldDir` includes Y, so the network is fed vertical error too; harmless, but the
  unit vector plus a *saturated* distance input means "how far" is unavailable.
- `period = 1.5 s` clock over a 6.67 s phase gives ~4.4 cycles — fine as an oscillator,
  but note it's the only phase-timing signal and it resets with `reset()`.

---

## 6. MEDIUM — the fitness graph is broken and hides stagnation

`Classes/UI/UiManager.cs:50-66`

```csharp
values: ref FitnessGraphValues.ToArray()[0],
```

`ToArray()` allocates a fresh array every frame; you take a `ref` into a temporary. With
ImGui.NET's `ref float` overload this does not compile; with a `float[]` overload it
passes a *copy*, so the plot never advances. Either way the graph is not reading the
live list. Additionally `scale_min: 0, scale_max: 100` is hardcoded while fitness is
unbounded — my replication of your objective produces scores of 240+, so a real run
saturates the graph and looks flat. **That is exactly what would hide "we stopped
improving at gen 300".**

Fix:

```csharp
float[] best = FitnessGraphValues.ToArray();      // one copy per frame is fine
ImGui.PlotLines("Best Fitness", ref best[0], best.Length, 0, "", float.MinValue, autoScaleMax, ...);
```

or keep a `float[]` buffer field that you append to. Compute `scale_max` from the data
(e.g. `MathF.Max(10f, FitnessGraphValues.Max() * 1.2f)`).

---

## 7. MEDIUM — selection-diversity mechanics

`Classes/NeuralNet/NNTrainer.cs:85-160`

- `poolSize = sorted.Count / 2` = **25**, and `SelectTournamentIndex` picks the **lowest
  index of 3 uniform draws** over `[0, 25)`. That is a fine rank-based tournament, but
  the pool *includes the 8 elites*, so elites are routinely re-copied as ordinary
  offspring. With 8 elites + 5 immigrants + ~7 crossover, ~30 of 50 slots per
  generation are near-duplicates of the top 8. Effective diversity is much lower than
  "50 creatures".
- `Pathological monotone mutation schedule` (`frac < 0.35 / 0.80 / else`) is decent, but
  all operators are Gaussian-on-weights only — no per-layer rates, no neuron-level
  add/remove, and `DefaultMutationRate = 0.2` applied uniformly to ~2000 weights gives
  ~400 edits/child. Large sigma (0.35) at 30% rate on 30% of children is a lot of
  churn for a fitness that is itself noisy across generations (targets are resampled
  every generation — see §8).
- Suggest: raise the tournament pool to the top ~70%, don't allow elites in the parent
  pool (or allow but cap their offspring), and reduce sigma on the top-of-rank
  children.

---

## 8. MEDIUM — fitness is not comparable across generations, which adds selection noise

Every generation resamples `baseAngle` and the shuffle, so generation *n* and *n+1* are
different tasks. Combined with the 5 immigrants and the non-monotone reward of §1, the
best-fitness curve is a random walk. That is legitimate for generalisation, but:

- It makes "the graph is flat" ambiguous between "no progress" and "noisy evaluation".
- `EvolveGeneration` doesn't use any of the history: `NNTrainer.Generations` is
  **never appended to**, and `LatestGen` is written but never read. So you have no
  in-memory record of past genomes to fall back to or inspect.
- `Objectives.MaxDistanceFromPopulation` is written in `EndOfGenerationTasks`
  (`RagdollBody.cs:385`) and **never read** — the relative-fitness mechanism it was
  built for is dead code (`float.MinValue` initial value and all).

Recommendation: evaluate each child on a **fixed bank of 2–3 target schedules** shared
across the generation (or seed the schedule RNG per generation and replay it), so scores
are comparable, then rotate a small fraction of the bank each generation for
generalisation. This alone makes the best-fitness curve meaningful.

---

## 9. LOW — reward-plumbing bugs

- `RagdollBody.cs:271-275`: `reached` is set once (`if (!reached && ...)`), and
  `TimeToReachTarget = clock` is the clock **since reset**, not since the phase started.
  So `EvalTimeTaken` (weight `0.5`) is a stale, cumulative quantity: reaching target 1
  at t=3 s gives the same bonus forever, and phases 2/3 never update it. Either
  reset `reached`/`TimeToReachTarget` per phase or drop the objective.
- `fitnessObjectives` weights (`RagdollBody.cs:65-70`): energy weight is `0f` and never
  enabled, so `EvalEnergyUsed`/`EnergyUsed` are maintained for nothing. If you ever turn
  it on, note `EnergyUsed += |output|` is a sum over *every* substep (~1200/phase), so
  `1/(1+E)` will be ~0 for every creature — it needs to be normalised by step count.
- `FitnessEval.cs:51`: `20f * EvalProgressToTarget(...)` in the no-phase path is a
  **20× scale jump** versus the 1× path. If a generation ever ends with zero finalised
  phases (pause, speed toggle, early break), that creature's score is inflated ~20×
  relative to everyone else in the same generation. Make both paths use the same scale.

---

## 10. LOW — bugs / dead code worth cleaning

- `NeuralNet/NeuralNetwork - extras.cs` — a second `NeuralNetwork` class in namespace
  `...NeuralNet.Extras`, unreferenced, with a broken dimension check
  (`Input.ColumnCount != LayerDims[0]`). Delete; it is a trap for future edits.
- `NeuralNetwork.ExecuteEndOfGenerationTasks` (`NeuralNetwork.cs:158-166`) builds
  `new Genome(clonedWeights, ConstructBiasList())` — `ConstructBiasList()` returns the
  **live `LayerBias` matrices**, not clones, so `genome.Biases` aliases the network.
  Nothing reads `genome` today, so it is harmless *now*, but any future crossover via
  `Genome` will silently corrupt parents. Clone them.
- `NeuralNetwork.Copy()` shares the `int[] LayerDims` reference between all copies.
  Harmless (never mutated) but worth cloning for safety.
- `NNTrainer.cs:3-5`: `using static System.ComponentModel.Design.ObjectSelectorEditor;`
  and `System.Windows.Forms.AxHost` are WinForms-designer leftovers.
- `Bone` constructor reloads and re-pointclouds `models/Raw/bone.stl` **per bone, per
  creature, per clone**. `Clone` is called 49× per generation. This is the real content
  of `todo.txt`'s "speed up training": cache the vertex list (and ideally the shape)
  statically.
- `Joint.cs` sets `Motor.MaximumForce = 400f` (800f in `initialiseConstraints`) on
  ~1 kg bones — that is a very high authority relative to the mass, so the motors can
  snap limbs to target velocity almost instantly and joints will mostly live at their
  ±45° limits. That tends to produce bang-bang, symmetric gaits rather than controlled
  turning. Worth logging motor saturation; if >90% of steps are at the limit, the
  network's fine-grained output is being thrown away.

---

## 11. Verify-only (I could not confirm, worth 5 minutes)

- `PhysicsObject.FromJquaternion` / `ToJQuaternion` negate `X` and `W` only
  (`(-x, y, z, -w)`) with the comment "some random bug requires negative". That is
  neither the identity nor the antipodal map, so the *rendered* orientation is a
  different rotation than the *simulated* one. Round-trips are consistent so nothing
  breaks, but the visual pose is not the physics pose.
- `Transform.CreateTransformationMatix` uses `Matrix4.CreateTranslation(-position.X, ...)`
  ("flipped x position so negative is left"), while `Camera` reportedly has
  `Right = (-1,0,0)`. Two X flips probably cancel and the view is correct — but if they
  don't, everything you see is mirrored in X, and "bottom left on screen" is a different
  physical corner than you think. Since `TargetCube` and the bones share the same visual
  path, a mirror would be consistent and harmless for learning, but harmful for
  interpreting your bird's-eye view. Quick check: press `P` (it prints COM in physics
  coordinates), walk a creature toward the on-screen bottom-left, and confirm the sign
  of X and Z in the console matches the screen.

---

## What I'd change first, in order

1. §1 fitness rewrite (relative, bounded, arrival-dominated). Non-negotiable — without
   it every other change just reshapes a treadmill.
2. §2 geometry: shrink the ring to ~12–18 m and/or lengthen phases; clamp radially.
   Without a reachable target, `PhasesReachedCount` stays 0 and arrivals never teach
   anything.
3. §3 make the target kinematic and out of creature collision; score the same position
   you steer to.
4. §4 fix the self-collision filter to only skip joint-adjacent pairs.
5. §5 replace `Bones[0]` with a proper body frame; un-saturate the distance input
   (`dist / 60f` or normalise by the current target distance).
6. §6/§8 fix the graph and pin the evaluation schedules, so you can *see* whether a
   change helped. Right now you are flying blind on the one plot you rely on.

Method note: findings in §1 come from a Monte-Carlo replication of
`initializeGenTargetSchedule` and the exact `EvalMultiPhaseTargetFollowing` formula
(4000 schedules per cell, 40 substeps/phase), plus a small GA using your
elite/tournament/Gaussian-mutation operators. §3–§10 are read directly off the source
and cross-checked against Jitter2 2.8.8 (`IBroadPhaseFilter`, `HingeAngle`, `BoxShape`,
`JVector.Transform`). The project has not been compiled or run here (no .NET SDK, and
it targets `net9.0-windows`).
