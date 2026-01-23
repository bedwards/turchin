"""Mathematical models for Structural-Demographic Theory."""

from cliodynamics.models.ages_of_discord import (
    AgesOfDiscordConfig,
    AgesOfDiscordModel,
    EliteWeights,
    PSIWeights,
    WellBeingWeights,
    compute_ages_of_discord_indices,
)
from cliodynamics.models.params import SDTParams, SDTState
from cliodynamics.models.sdt import SDTModel

__all__ = [
    # SDT Model
    "SDTParams",
    "SDTState",
    "SDTModel",
    # Ages of Discord Model
    "AgesOfDiscordModel",
    "AgesOfDiscordConfig",
    "WellBeingWeights",
    "EliteWeights",
    "PSIWeights",
    "compute_ages_of_discord_indices",
]
