# deep-belief-betting

`deep-belief-betting` is a research-oriented deep reinforcement learning codebase for studying **single-roundtrip trading in binary prediction markets** under **partial observability**, **LMSR market making**, and **reinforcement learning**.

The project is built around a clean decomposition:

1. **Market simulation**
   Generate realistic path data for a binary prediction market with a hidden latent belief process, informed and noise trader flow, and endogenous public prices through LMSR
2. **Belief pretraining**
   Learn predictive state representations from simulated paths using supervised sequence learning
3. **Reinforcement learning for roundtrip timing**
   Solve the control problem of when to enter a YES or NO position and when to exit, under a strict single-roundtrip constraint

The core idea is that this is **not just a forecasting problem**. The agent interacts with the market, its trades move the LMSR state, and the value of acting now must be compared against the value of waiting. That makes the problem a **stochastic optimal control problem with nested optimal stopping structure**.

---

## project motivation

Binary prediction markets are attractive because the object being traded is simple, but the decision problem is not.

Even in a two-outcome market, a trader still faces several hard questions:

- is the current public probability mispriced relative to private belief
- is it better to enter now or wait for more information
- once invested, is it better to hold or unwind
- how should market impact and transaction costs change timing decisions

This repo focuses on the **simplest nontrivial version** of that problem:

- one market
- one binary event
- one agent
- one fixed-size trade
- at most one entry
- at most one exit
- no re-entry

That restriction is deliberate. It keeps the problem economically meaningful while making the control structure interpretable and implementable.

---

## problem setup

The project studies a **single-roundtrip betting problem** in a binary LMSR prediction market.

The agent begins flat and at each decision time can:

- wait
- buy YES
- buy NO
- if already invested, sell its current position

The implementation uses a **fixed trade size** relative to the LMSR liquidity parameter `b`. This keeps the action space discrete and stable for a proof-of-concept RL system.

The market maker follows the **logarithmic market scoring rule**. In the binary case, the market state can be represented using a single net YES inventory variable \(Q_t\), and the public YES probability is the corresponding LMSR quote. Your underlying project note uses exactly this representation and frames the agent’s economics through execution cost and liquidation value under LMSR. :contentReference[oaicite:1]{index=1}

---

## why this is an optimal stopping problem

This is not just a classification task and not just a sequence modelling task.

While the agent is **flat**, it faces an **entry stopping problem**:

- continue waiting
- or stop waiting and enter YES
- or stop waiting and enter NO

Once the agent is **invested**, it faces an **exit stopping problem**:

- continue holding
- or stop holding and unwind now

That creates a natural **nested stopping structure**:

- flat regime
  compare waiting versus entering
- invested regime
  compare holding versus exiting

The project note formalises this structure explicitly through flat, long-YES, and long-NO value functions and corresponding Bellman recursions. :contentReference[oaicite:2]{index=2}

This matters because the central object is not merely whether the event resolves YES or NO. The central object is the **continuation value of waiting** versus the **realised value of trading now**.

---

## project hierarchy

The repo is organised around a three-level hierarchy.

### problem 1: inference

Infer a predictive belief about terminal resolution from the observed market path.

This is the supervised learning layer. Given a sequence of market observations, learn a representation that helps estimate terminal outcome.

### problem 2: roundtrip control

Use that predictive representation to decide:

- when to enter
- which side to enter
- when to exit

This is the main RL problem addressed by the repo.

### problem 3: richer position management

Allow dynamic sizing, scaling in, scaling out, and richer inventory control.

This is explicitly out of scope for the current implementation, but the architecture is designed so that a later version can add learned trade size and more flexible action spaces.

## market simulator

The simulator is designed to produce paths that are rich enough to make the control problem meaningful, but still simple enough to reason about and code carefully.

### simulator design principles

The simulator includes:

- a **latent belief process** in log-odds space
- a **noisy private signal** observed by informed traders
- a **filtered informed execution-pressure state**
- a **persistent noise-pressure state** for clustered uninformed flow
- **endogenous public probability** generated through a binary LMSR

This design is important because public prices should not be a trivial transform of hidden truth. The simulator should produce markets where information is revealed **noisily**, **gradually**, and **path-dependently** rather than instantly. 

### time-dependent terminal anchoring

The hidden process is not pulled directly toward a hard terminal label from the start of the episode.

Instead, the simulator uses:

- a noisy terminal conviction anchor
- a maturity-dependent anchoring schedule
- regime-dependent baseline levels

This weakens early path separation and makes the filtering problem less trivial.

### order flow decomposition

Total signed order flow is the sum of:

- **informed flow**
- **noise flow**

Both are modelled separately so that the market contains both informational and non-informational movement. This separation is central to the simulator and is one of the reasons the belief model can be meaningful.

---

## learning setup

The repo supports a two-stage pipeline.

### stage 1: supervised pretraining

The simulator can generate path-level datasets for sequence models.

Default use case:

- input
  simulated market path features
- main target
  terminal resolution label

Optional auxiliary targets can include next-step observed quantities such as:

- next public probability
- next-step flow sign

The goal is not to recover the true hidden simulator state in a structural sense. The goal is to learn a **predictive state representation** that is useful online.

### stage 2: reinforcement learning

The RL agent operates in a Gymnasium environment built on top of:

- `MarketSim` for exogenous market dynamics
- `Broker` for agent-side accounting and execution economics
- `PredictionMarketEnv` for the RL interface

The agent can consume:

- no belief features
- a pretrained belief vector

This makes it easy to run ablations and test whether learned belief actually improves stopping decisions.

---

## reward design

The environment supports multiple reward specifications for ablation.

### preferred objective

The preferred objective is:

- **terminal net episode PnL**

This keeps the optimisation target aligned with the economics of the roundtrip problem and avoids injecting unnecessary human-designed preferences into the policy.

### alternative reward mode

The environment can also support:

- **realised cashflow style rewards**

This is useful for comparison, debugging, or experimentation, but it is not the preferred objective.

---

## action space and masking

The current implementation uses a small masked discrete action space.

Action indices:

- `0` hold or wait
- `1` buy YES
- `2` buy NO
- `3` sell current position

Masks enforce the single-roundtrip structure:

- if flat
  hold, buy YES, buy NO are valid
- if invested
  hold and sell are valid
- if dead post-exit state
  only hold is valid

This keeps the environment mathematically faithful to the stopping structure while staying simple enough for PyTorch and Gym models. 

---

## observation design

The environment observation is designed to expose the right local state without excessive manual feature engineering.

Typical observation components include:

- public probability
- same-day net order flow
- time to resolution
- has-entered or has-position flag
- position side
- realised cash PnL so far
- entry public probability
- broker-computed entry economics when flat
- broker-computed unwind value when invested
- optional pretrained belief vector
- explicit dead-state flag

Two design decisions are important here.

### 1. same-day flow only

The environment uses **same-day order flow** rather than a long manually engineered history stack. That reflects the project preference for minimal hand-crafted features and stronger reliance on learning and compute.

### 2. broker-computed LMSR economics

The broker exposes local execution and unwind quantities directly. This is deliberate. The agent should not be forced to rediscover the LMSR accounting identity from raw public probability alone.

---

## code structure

The intended package layout is:

```text
deep-belief-betting/
├── configs/
│   └── default.yaml
├── notebooks/
│   └── smoke-test.ipynb
├── src/
│   └── deep_belief_betting/
│       ├── __init__.py
│       ├── parameters.py
│       ├── market_sim.py
│       ├── broker.py
│       ├── prediction_market_env.py
│       ├── pretraining_path_generator.py
│       ├── env_factory.py
│       └── smoke_train.py
├── tests/
│   ├── test_parameters.py
│   ├── test_market_sim.py
│   ├── test_broker.py
│   ├── test_env.py
│   └── test_pretraining_generator.py
├── pyproject.toml
└── README.md
```
