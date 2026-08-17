"""MGC underwater flexible optical-cable deployment simulator."""

from .config import SimulationConfig, load_config
from .simulation import run_simulation

__all__ = ["SimulationConfig", "load_config", "run_simulation"]
__version__ = "0.1.0"
