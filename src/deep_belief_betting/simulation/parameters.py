from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import yaml


class ParameterValidationError(ValueError):
    """Raised when a configuration file is invalid."""


def default_trade_size_policy(b: float) -> int:
    """Map LMSR liquidity to a fixed integer trade size."""
    return round(0.10 * b)


@dataclass(frozen=True)
class InitialStateConfig:
    """Initial market state."""

    x0: float
    g0: float
    noise_pressure0: float
    q0: float


@dataclass(frozen=True)
class TerminalConfig:
    """Terminal outcome controls."""

    epsilon: float
    yes_probability: float


@dataclass(frozen=True)
class TerminalConvictionConfig:
    """Noisy terminal anchor controls."""

    alpha_yes: float
    beta_yes: float
    alpha_no: float
    beta_no: float


@dataclass(frozen=True)
class AnchorScheduleConfig:
    """Time-varying anchor schedule."""

    schedule_type: str
    lambda_start: float
    lambda_end: float
    lambda_power: float


@dataclass(frozen=True)
class RegimeConfig:
    """Regime process controls."""

    theta_bar: list[float]
    initial_probs: list[float]
    transition_matrix: list[list[float]]


@dataclass(frozen=True)
class LatentBeliefConfig:
    """Latent belief diffusion controls."""

    kappa: float
    sigma: float


@dataclass(frozen=True)
class JumpConfig:
    """Jump process controls."""

    lambda_j: float
    eta_0: float
    eta_1: float
    sigma_j: float


@dataclass(frozen=True)
class PrivateSignalConfig:
    """Private information noise."""

    sigma_sig: float


@dataclass(frozen=True)
class ExecutionPressureConfig:
    """Filtered informed pressure controls."""

    rho: float
    sigma_g: float


@dataclass(frozen=True)
class InformedFlowConfig:
    """Informed order flow controls."""

    pi_min_inf: float
    pi_max_inf: float
    gamma_pi: float
    beta_inf: float
    m_inf: float


@dataclass(frozen=True)
class NoiseFlowConfig:
    """Noise trader flow controls."""

    pi_noise: float
    phi_noise_pressure: float
    sigma_noise_pressure: float
    beta_noise: float
    m_noise: float


@dataclass(frozen=True)
class LMSRConfig:
    """LMSR pricing controls."""

    b: float


@dataclass(frozen=True)
class TradeConfig:
    """Trading controls."""

    allow_reentry: bool
    trade_size_policy_name: str
    terminate_on_exit: bool


@dataclass(frozen=True)
class FeeConfig:
    """Explicit execution fees."""

    fixed_fee: float
    proportional_fee_bps: float


@dataclass(frozen=True)
class RewardConfig:
    """Reward controls."""

    reward_mode: str
    inactivity_penalty: float
    invalid_action_penalty: float


@dataclass(frozen=True)
class BeliefFeatureConfig:
    """Belief feature controls."""

    enabled: bool
    mode: str


@dataclass(frozen=True)
class FeatureConfig:
    """Observation feature toggles."""

    include_public_probability: bool
    include_same_day_flow: bool
    include_time_to_resolution: bool
    include_position_flag: bool
    include_position_side: bool
    include_cash_pnl: bool
    include_entry_probability: bool
    include_entry_costs_when_flat: bool
    include_unwind_value_when_invested: bool
    include_belief_features: bool
    explicit_dead_state: bool
    include_belief_q: bool = False
    include_belief_terminal: bool = False


@dataclass(frozen=True)
class ObservationNormalizationConfig:
    """Observation normalization controls."""

    enabled: bool
    clip_value: float


@dataclass(frozen=True)
class Parameters:
    """Top-level immutable project configuration."""

    seed: int
    horizon_days: int
    time_step_days: float
    initial_state: InitialStateConfig
    terminal: TerminalConfig
    terminal_conviction: TerminalConvictionConfig
    anchor_schedule: AnchorScheduleConfig
    regimes: RegimeConfig
    latent_belief: LatentBeliefConfig
    jump: JumpConfig
    private_signal: PrivateSignalConfig
    execution_pressure: ExecutionPressureConfig
    informed_flow: InformedFlowConfig
    noise_flow: NoiseFlowConfig
    lmsr: LMSRConfig
    trade: TradeConfig
    fees: FeeConfig
    reward: RewardConfig
    belief_features: BeliefFeatureConfig
    features: FeatureConfig
    observation_normalization: ObservationNormalizationConfig

    @property
    def num_steps(self) -> int:
        """Return the number of discrete simulation steps."""
        if self.horizon_days <= 0:
            raise ParameterValidationError("horizon_days must be positive.")
        if self.time_step_days <= 0.0:
            raise ParameterValidationError("time_step_days must be positive.")
        ratio = self.horizon_days / self.time_step_days
        if abs(ratio - round(ratio)) > 1e-9:
            raise ParameterValidationError("horizon_days must be divisible by time_step_days.")
        return int(round(ratio))

    @property
    def dt(self) -> float:
        """Return the simulation time increment."""
        return self.time_step_days

    def trade_size(self) -> int:
        """Return the fixed agent trade size implied by LMSR liquidity."""
        if self.trade.trade_size_policy_name != "default_fraction_of_b":
            raise ParameterValidationError(
                f"Unknown trade size policy: {self.trade.trade_size_policy_name}"
            )

        size = default_trade_size_policy(self.lmsr.b)
        if size <= 0:
            raise ParameterValidationError("trade size policy must return a positive integer.")
        return size

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Parameters":
        """Load parameters from a YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"YAML file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        if not isinstance(raw, Mapping):
            raise ParameterValidationError("Top-level YAML must be a mapping.")

        return cls.from_dict(dict(raw))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Parameters":
        """Build parameters from a nested dictionary."""
        required_top_keys = {
            "seed",
            "horizon_days",
            "time_step_days",
            "initial_state",
            "terminal",
            "terminal_conviction",
            "anchor_schedule",
            "regimes",
            "latent_belief",
            "jump",
            "private_signal",
            "execution_pressure",
            "informed_flow",
            "noise_flow",
            "lmsr",
            "trade",
            "fees",
            "reward",
            "belief_features",
            "features",
            "observation_normalization",
        }
        missing = required_top_keys.difference(raw)
        if missing:
            raise ParameterValidationError(f"Missing required keys: {sorted(missing)}")

        params = cls(
            seed=int(raw["seed"]),
            horizon_days=int(raw["horizon_days"]),
            time_step_days=float(raw["time_step_days"]),
            initial_state=InitialStateConfig(
                x0=float(raw["initial_state"]["x0"]),
                g0=float(raw["initial_state"]["g0"]),
                noise_pressure0=float(raw["initial_state"]["noise_pressure0"]),
                q0=float(raw["initial_state"]["q0"]),
            ),
            terminal=TerminalConfig(
                epsilon=float(raw["terminal"]["epsilon"]),
                yes_probability=float(raw["terminal"]["yes_probability"]),
            ),
            terminal_conviction=TerminalConvictionConfig(
                alpha_yes=float(raw["terminal_conviction"]["alpha_yes"]),
                beta_yes=float(raw["terminal_conviction"]["beta_yes"]),
                alpha_no=float(raw["terminal_conviction"]["alpha_no"]),
                beta_no=float(raw["terminal_conviction"]["beta_no"]),
            ),
            anchor_schedule=AnchorScheduleConfig(
                schedule_type=str(raw["anchor_schedule"]["schedule_type"]),
                lambda_start=float(raw["anchor_schedule"]["lambda_start"]),
                lambda_end=float(raw["anchor_schedule"]["lambda_end"]),
                lambda_power=float(raw["anchor_schedule"]["lambda_power"]),
            ),
            regimes=RegimeConfig(
                theta_bar=[float(x) for x in raw["regimes"]["theta_bar"]],
                initial_probs=[float(x) for x in raw["regimes"]["initial_probs"]],
                transition_matrix=[
                    [float(x) for x in row]
                    for row in raw["regimes"]["transition_matrix"]
                ],
            ),
            latent_belief=LatentBeliefConfig(
                kappa=float(raw["latent_belief"]["kappa"]),
                sigma=float(raw["latent_belief"]["sigma"]),
            ),
            jump=JumpConfig(
                lambda_j=float(raw["jump"]["lambda_j"]),
                eta_0=float(raw["jump"]["eta_0"]),
                eta_1=float(raw["jump"]["eta_1"]),
                sigma_j=float(raw["jump"]["sigma_j"]),
            ),
            private_signal=PrivateSignalConfig(
                sigma_sig=float(raw["private_signal"]["sigma_sig"])
            ),
            execution_pressure=ExecutionPressureConfig(
                rho=float(raw["execution_pressure"]["rho"]),
                sigma_g=float(raw["execution_pressure"]["sigma_g"]),
            ),
            informed_flow=InformedFlowConfig(
                pi_min_inf=float(raw["informed_flow"]["pi_min_inf"]),
                pi_max_inf=float(raw["informed_flow"]["pi_max_inf"]),
                gamma_pi=float(raw["informed_flow"]["gamma_pi"]),
                beta_inf=float(raw["informed_flow"]["beta_inf"]),
                m_inf=float(raw["informed_flow"]["m_inf"]),
            ),
            noise_flow=NoiseFlowConfig(
                pi_noise=float(raw["noise_flow"]["pi_noise"]),
                phi_noise_pressure=float(raw["noise_flow"]["phi_noise_pressure"]),
                sigma_noise_pressure=float(raw["noise_flow"]["sigma_noise_pressure"]),
                beta_noise=float(raw["noise_flow"]["beta_noise"]),
                m_noise=float(raw["noise_flow"]["m_noise"]),
            ),
            lmsr=LMSRConfig(
                b=float(raw["lmsr"]["b"])
            ),
            trade=TradeConfig(
                allow_reentry=bool(raw["trade"]["allow_reentry"]),
                trade_size_policy_name=str(raw["trade"]["trade_size_policy_name"]),
                terminate_on_exit=bool(raw["trade"]["terminate_on_exit"]),
            ),
            fees=FeeConfig(
                fixed_fee=float(raw["fees"]["fixed_fee"]),
                proportional_fee_bps=float(raw["fees"]["proportional_fee_bps"]),
            ),
            reward=RewardConfig(
                reward_mode=str(raw["reward"]["reward_mode"]),
                inactivity_penalty=float(raw["reward"]["inactivity_penalty"]),
                invalid_action_penalty=float(raw["reward"]["invalid_action_penalty"]),
            ),
            belief_features=BeliefFeatureConfig(
                enabled=bool(raw["belief_features"]["enabled"]),
                mode=str(raw["belief_features"]["mode"]),
            ),
            features=FeatureConfig(
                include_public_probability=bool(raw["features"]["include_public_probability"]),
                include_same_day_flow=bool(raw["features"]["include_same_day_flow"]),
                include_time_to_resolution=bool(raw["features"]["include_time_to_resolution"]),
                include_position_flag=bool(raw["features"]["include_position_flag"]),
                include_position_side=bool(raw["features"]["include_position_side"]),
                include_cash_pnl=bool(raw["features"]["include_cash_pnl"]),
                include_entry_probability=bool(raw["features"]["include_entry_probability"]),
                include_entry_costs_when_flat=bool(raw["features"]["include_entry_costs_when_flat"]),
                include_unwind_value_when_invested=bool(raw["features"]["include_unwind_value_when_invested"]),
                include_belief_features=bool(raw["features"]["include_belief_features"]),
                explicit_dead_state=bool(raw["features"]["explicit_dead_state"]),
                include_belief_q=bool(raw["features"].get("include_belief_q", False)),
                include_belief_terminal=bool(raw["features"].get("include_belief_terminal", False)),
            ),
            observation_normalization=ObservationNormalizationConfig(
                enabled=bool(raw["observation_normalization"]["enabled"]),
                clip_value=float(raw["observation_normalization"]["clip_value"]),
            ),
        )

        params.validate()
        return params

    def validate(self) -> None:
        """Validate configuration consistency."""
        if not (0.0 < self.terminal.epsilon < 0.5):
            raise ParameterValidationError("terminal epsilon must lie in open interval zero to one half")

        if not (0.0 < self.terminal.yes_probability < 1.0):
            raise ParameterValidationError("terminal yes probability must lie in open interval zero to one")

        if self.anchor_schedule.schedule_type.lower() != "linear":
            raise ParameterValidationError("only linear anchor schedule is supported in v2")

        if not (0.0 <= self.anchor_schedule.lambda_start <= 1.0):
            raise ParameterValidationError("lambda start must lie in closed interval zero to one")

        if not (0.0 <= self.anchor_schedule.lambda_end <= 1.0):
            raise ParameterValidationError("lambda end must lie in closed interval zero to one")

        if self.anchor_schedule.lambda_end < self.anchor_schedule.lambda_start:
            raise ParameterValidationError("lambda end must be at least lambda start")

        if self.anchor_schedule.lambda_power <= 0.0:
            raise ParameterValidationError("lambda power must be positive")

        if self.jump.lambda_j * self.dt > 1.0:
            raise ParameterValidationError("jump probability per step must not exceed one")

        if not (0.0 < self.execution_pressure.rho < 1.0):
            raise ParameterValidationError("execution pressure rho must lie in open interval zero to one")

        if not (0.0 < self.noise_flow.pi_noise < 1.0):
            raise ParameterValidationError("noise participation must lie in open interval zero to one")

        if not (0.0 <= self.noise_flow.phi_noise_pressure < 1.0):
            raise ParameterValidationError("noise pressure persistence must lie in half open interval zero to one")

        if not (0.0 <= self.informed_flow.pi_min_inf <= self.informed_flow.pi_max_inf <= 1.0):
            raise ParameterValidationError("informed participation bounds are invalid")

        if self.informed_flow.m_inf <= 0.0 or self.noise_flow.m_noise <= 0.0:
            raise ParameterValidationError("trade size means must be positive")

        if self.lmsr.b <= 0.0:
            raise ParameterValidationError("lmsr liquidity b must be positive")

        if self.reward.reward_mode not in {"terminal_net_pnl", "realized_cashflow"}:
            raise ParameterValidationError("unsupported reward mode")

        if self.belief_features.mode not in {"none", "vector"}:
            raise ParameterValidationError("belief feature mode must be none or vector")

        if self.trade.terminate_on_exit:
            raise ParameterValidationError("v2 keeps running after exit to preserve explicit dead state")
  
        if self.observation_normalization.clip_value <= 0.0:
            raise ParameterValidationError("observation normalization clip value must be positive")
