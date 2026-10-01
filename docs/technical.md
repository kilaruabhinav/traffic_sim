# Technical Design

This document describes the implementation behind the traffic-control experiment. The
source code remains authoritative.

## 1. Environment lifecycle

`TrafficEnv` implements the Gymnasium API. `reset()` clears all queues and episode
counters, chooses lane 1 as the initial green lane unless an internal option overrides
it, and attaches the environment's entropy-backed NumPy generator to `TrafficSystem`.
Calling `reset()` without a seed advances the existing random stream across episodes.

For one `step(action)`, the environment:

1. records the previous normalized queue pressure;
2. advances signal timing and decides whether discharge is allowed;
3. applies the step's generated or replayed attempted arrivals;
4. increments aggregate waiting for every queued vehicle;
5. discharges up to two vehicles from the active lane when permitted;
6. updates throughput, queue-limit, and maximum-queue counters;
7. calculates bounded reward components;
8. constructs the next 15-value observation and diagnostic `info`; and
9. truncates at 600 steps.

The environment never emits `terminated=True`; the fixed horizon uses Gymnasium's
`truncated` flag.

## 2. Traffic arrival model

Each lane independently attempts one arrival per step with Bernoulli probability 0.35.
An attempt is accepted unless the lane is already at its hard capacity of 40. Rejected
attempts increment `dropped_arrivals`.

Normal episodes draw directly from the environment RNG. Evaluation instead creates an
immutable `TrafficScenario` containing a 600×4 Boolean attempted-arrival matrix. The
same object is replayed for every controller in a scenario.

## 3. Queue model

Each `LaneTraffic` stores an integer queue length and aggregate waiting. Arrivals join
the queue before waiting is incremented, so every queued vehicle contributes one unit
that step. Discharge removes at most two vehicles and proportionally removes their
share of aggregate waiting, preserving the lane's aggregate mean-wait approximation.
Queues cannot exceed 40 or become negative.

The soft limit of 20 affects metrics and the overflow reward term; it does not block
arrivals. A violation is counted once per lane-step where queue length exceeds 20.

## 4. Signal switching model

Exactly one lane is represented as active green. A request for another lane cannot
start a switch until the current green time reaches 3. A valid request starts one
clearance step in which no lane discharges; the pending lane then becomes active. If
green time reaches 12 while the controller keeps requesting the active lane, control is
forced to the next lane cyclically.

## 5. Observation construction

The `float32` observation contains 15 values in `[0,1]`:

1. four queue lengths divided by their hard limits;
2. four aggregate mean waits divided by 30 and clipped;
3. four one-hot current-green indicators;
4. green time divided by 12 and clipped;
5. one switching/clearance flag; and
6. the zero-based pending lane divided by 3.

Pending lane is zero when absent, so the switching flag distinguishes absence from a
pending lane-1 transition.

## 6. Reward calculation

The reward combines normalized throughput, queue pressure, clipped waiting pressure,
quadratic overflow above the soft limit, clipped starvation above mean wait 15, queue
pressure progress, and a -0.02 switch-initiation cost. Coefficients are defined in
`TrafficConfig`; the complete equations are in the README. Every component is copied
to `info["reward_components"]` for debugging.

## 7. DQN training

`train_dqn()` validates the environment, wraps it with an SB3 `Monitor`, creates a
64×64 `MlpPolicy` DQN, trains for the configured budget, and saves both the model and a
JSON configuration. The current configuration uses a 50,000-transition replay buffer,
1,000 learning-start steps, batch size 64, gamma 0.99, training every four environment
steps, one gradient step, a 1,000-step target update, and epsilon decay over 40% of the
run to 0.05.

## 8. PPO training

`train_ppo()` uses an SB3 actor-critic `MlpPolicy` with separate 64×64 policy and value
networks. Rollouts contain 2,048 steps and are optimized for 10 epochs in batches of 64
with gamma 0.99, GAE lambda 0.95, clip range 0.2, and entropy coefficient 0.01. Passing
`--resume-from` loads a PPO archive and treats `--timesteps` as additional training.

Neither normal training path supplies a fixed SB3 seed.

## 9. Evaluation methodology

The evaluator generates scenarios in an outer loop and controllers in an inner loop,
ensuring paired traffic demand. Deterministic model predictions are used; the Random
baseline retains independent action randomness. Per-episode records include reward,
the established waiting metric, maximum queue, throughput, violations, and limit flags.

Summaries use NumPy means, medians, and sample standard deviations (`ddof=1`). Mean
confidence intervals use the normal approximation
`mean ± 1.96 × sample_sd / sqrt(n)`. PPO-minus-DQN values are calculated per scenario
before applying the same summary formula. The JSON writer stores raw scenarios and
results before the plotting script reads them.

## 10. GUI architecture

`TrafficVisualizer` loads DQN or PPO according to `--algorithm` and calls deterministic
`predict`. It never changes numerical state directly. After every environment step, it
mirrors accepted arrivals and creates one `MovingVehicle` for every reported departure.
Queue identities are removed only when their individual departure animations finish.
Tkinter owns rendering and timing; `TrafficEnv` remains authoritative for signals,
queues, reward, throughput, and violations.
