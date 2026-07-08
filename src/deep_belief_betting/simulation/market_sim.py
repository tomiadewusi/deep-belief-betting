from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from deep_belief_betting.simulation.parameters import Parameters


@dataclass(frozen=True)
class MarketState:
    """Current exogenous market state."""

    step: int
    time_to_resolution: int
    x: float
    g: float
    noise_pressure: float
    q: float
    regime: int
    theta_star: float
    terminal_outcome: int
    public_probability: float
    private_probability: float
    latent_probability: float
    delta_q: float
    informed_flow: float
    noise_flow: float


class MarketSim:
    """Discrete-time exogenous market simulator."""

    def __init__(self, params: Parameters):
        self.params = params
        self.rng = np.random.default_rng(params.seed)

        self._state: Optional[MarketState] = None
        self._regime_path: Optional[np.ndarray] = None
        self._done: bool = False

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _logit(p: float) -> float:
        """Numerically stable logit."""
        p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
        return float(np.log(p / (1.0 - p)))

    @staticmethod
    def _sample_positive_poisson(mean: float, rng: np.random.Generator) -> int:
        """Sample a Poisson draw conditioned on being positive."""
        draw = 0
        while draw < 1:
            draw = int(rng.poisson(mean))
        return draw

    def _anchor_weight(self, step: int) -> float:
        """Compute time-dependent terminal anchor weight."""
        frac = step / self.params.num_steps
        anchor = (
            self.params.anchor_schedule.lambda_start
            + (self.params.anchor_schedule.lambda_end - self.params.anchor_schedule.lambda_start)
            * (frac ** self.params.anchor_schedule.lambda_power)
        )
        return float(np.clip(anchor, 0.0, 1.0))

    def _sample_terminal_outcome(self) -> int:
        """Sample terminal YES or NO outcome."""
        return int(self.rng.binomial(1, self.params.terminal.yes_probability))

    def _sample_theta_star(self, terminal_outcome: int) -> float:
        """Sample noisy terminal anchor in log odds space."""
        tc = self.params.terminal_conviction

        if terminal_outcome == 1:
            beta_draw = self.rng.beta(tc.alpha_yes, tc.beta_yes)
            p_star = 0.5 + 0.5 * beta_draw
        else:
            beta_draw = self.rng.beta(tc.alpha_no, tc.beta_no)
            p_star = 0.5 * beta_draw

        p_star = float(np.clip(p_star, self.params.terminal.epsilon, 1.0 - self.params.terminal.epsilon))
        return self._logit(p_star)

    def _sample_regime_path(self) -> np.ndarray:
        """Sample the full latent regime path for one episode."""
        rg = self.params.regimes
        num_regimes = len(rg.theta_bar)

        path = np.zeros(self.params.num_steps + 1, dtype=np.int64)
        path[0] = int(self.rng.choice(num_regimes, p=np.asarray(rg.initial_probs, dtype=float)))

        transition = np.asarray(rg.transition_matrix, dtype=float)
        for t in range(self.params.num_steps):
            path[t + 1] = int(self.rng.choice(num_regimes, p=transition[path[t]]))

        return path

    def _theta_n(self, step: int, theta_star: float, regime: int) -> float:
        """Compute the time varying mean reversion level."""
        lam = self._anchor_weight(step)
        theta_bar = self.params.regimes.theta_bar[regime]
        return lam * theta_star + (1.0 - lam) * theta_bar

    def reset(self, seed: Optional[int] = None) -> MarketState:
        """Reset the simulator and return the initial market state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        terminal_outcome = self._sample_terminal_outcome()
        theta_star = self._sample_theta_star(terminal_outcome)
        self._regime_path = self._sample_regime_path()
        self._done = False

        init = self.params.initial_state
        public_probability = self._sigmoid(init.q0 / self.params.lmsr.b)
        private_probability = self._sigmoid(init.x0)
        latent_probability = self._sigmoid(init.x0)

        self._state = MarketState(
            step=0,
            time_to_resolution=self.params.num_steps,
            x=init.x0,
            g=init.g0,
            noise_pressure=init.noise_pressure0,
            q=init.q0,
            regime=int(self._regime_path[0]),
            theta_star=theta_star,
            terminal_outcome=terminal_outcome,
            public_probability=public_probability,
            private_probability=private_probability,
            latent_probability=latent_probability,
            delta_q=0.0,
            informed_flow=0.0,
            noise_flow=0.0,
        )
        return self._state

    def get_state(self) -> MarketState:
        """Return the current market state."""
        if self._state is None:
            raise RuntimeError("simulator must be reset before use")
        return self._state

    def apply_agent_trade(self, signed_yes_contracts: float) -> None:
        """Apply the agent trade to LMSR inventory before exogenous flow."""
        if self._state is None:
            raise RuntimeError("simulator must be reset before use")

        # update public inventory before the market moves
        new_q = self._state.q + signed_yes_contracts
        new_public_probability = self._sigmoid(new_q / self.params.lmsr.b)

        self._state = MarketState(
            step=self._state.step,
            time_to_resolution=self._state.time_to_resolution,
            x=self._state.x,
            g=self._state.g,
            noise_pressure=self._state.noise_pressure,
            q=new_q,
            regime=self._state.regime,
            theta_star=self._state.theta_star,
            terminal_outcome=self._state.terminal_outcome,
            public_probability=new_public_probability,
            private_probability=self._state.private_probability,
            latent_probability=self._state.latent_probability,
            delta_q=self._state.delta_q,
            informed_flow=self._state.informed_flow,
            noise_flow=self._state.noise_flow,
        )

    def step(self) -> tuple[MarketState, bool]:
        """Advance the exogenous market by one day."""
        if self._state is None:
            raise RuntimeError("simulator must be reset before use")
        if self._done:
            raise RuntimeError("cannot step a finished simulator")
        if self._regime_path is None:
            raise RuntimeError("regime path must be initialised")

        s = self._state
        p = self.params

        current_step = s.step
        next_step = current_step + 1

        regime_next = int(self._regime_path[next_step])
        theta_n = self._theta_n(current_step, s.theta_star, s.regime)

        # latent jump term
        jump_occurs = int(self.rng.binomial(1, p.jump.lambda_j * p.dt))
        eta_n = p.jump.eta_0 + p.jump.eta_1 * (p.horizon_days - current_step * p.dt)
        jump_size = eta_n * (s.theta_star - s.x) + p.jump.sigma_j * self.rng.normal()
        jump_term = jump_occurs * jump_size

        # hidden belief update
        x_next = (
            s.x
            + p.latent_belief.kappa * (theta_n - s.x) * p.dt
            + p.latent_belief.sigma * np.sqrt(p.dt) * self.rng.normal()
            + jump_term
        )

        # informed signal and pressure
        private_signal = s.x + p.private_signal.sigma_sig * self.rng.normal()
        private_probability = self._sigmoid(private_signal)
        g_next = (
            p.execution_pressure.rho * s.g
            + (1.0 - p.execution_pressure.rho) * (private_probability - s.public_probability)
            + p.execution_pressure.sigma_g * self.rng.normal()
        )

        # informed flow
        pi_inf = (
            p.informed_flow.pi_min_inf
            + (p.informed_flow.pi_max_inf - p.informed_flow.pi_min_inf)
            * self._sigmoid(p.informed_flow.gamma_pi * abs(s.g))
        )
        informed_active = int(self.rng.binomial(1, pi_inf))
        if informed_active == 1:
            yes_prob_inf = self._sigmoid(p.informed_flow.beta_inf * s.g)
            informed_direction = 1 if self.rng.uniform() < yes_prob_inf else -1
            informed_size = self._sample_positive_poisson(p.informed_flow.m_inf, self.rng)
        else:
            informed_direction = 0
            informed_size = 0
        informed_flow = float(informed_direction * informed_size)

        # clustered noise flow
        noise_pressure_next = (
            p.noise_flow.phi_noise_pressure * s.noise_pressure
            + p.noise_flow.sigma_noise_pressure * self.rng.normal()
        )
        noise_active = int(self.rng.binomial(1, p.noise_flow.pi_noise))
        if noise_active == 1:
            yes_prob_noise = self._sigmoid(p.noise_flow.beta_noise * s.noise_pressure)
            noise_direction = 1 if self.rng.uniform() < yes_prob_noise else -1
            noise_size = self._sample_positive_poisson(p.noise_flow.m_noise, self.rng)
        else:
            noise_direction = 0
            noise_size = 0
        noise_flow = float(noise_direction * noise_size)

        # exogenous market flow
        delta_q = informed_flow + noise_flow
        q_next = s.q + delta_q
        public_probability_next = self._sigmoid(q_next / p.lmsr.b)

        self._done = next_step >= p.num_steps
        self._state = MarketState(
            step=next_step,
            time_to_resolution=p.num_steps - next_step,
            x=x_next,
            g=g_next,
            noise_pressure=noise_pressure_next,
            q=q_next,
            regime=regime_next,
            theta_star=s.theta_star,
            terminal_outcome=s.terminal_outcome,
            public_probability=public_probability_next,
            private_probability=private_probability,
            latent_probability=self._sigmoid(x_next),
            delta_q=delta_q,
            informed_flow=informed_flow,
            noise_flow=noise_flow,
        )
        return self._state, self._done