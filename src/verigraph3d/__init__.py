"""VeriGraph3D research prototype."""

from .agent import VeriGraph3DAgent
from .agentblender import (
    AgentBlenderAtomicBlendBackend,
    AgentBlenderStateMapper,
    AgentBlenderRuntimeBridge,
    AgentBlenderTransactionBackend,
    AgentBlenderWorldStateBackend,
)
from .models import GoalSpec, SceneState, SemanticAction
from .vlm import VLMTaskInterpreter, VLMVisualVerifier, create_vlm_client

__all__ = [
    "GoalSpec", "SceneState", "SemanticAction", "VeriGraph3DAgent",
    "VLMTaskInterpreter", "VLMVisualVerifier", "create_vlm_client",
    "AgentBlenderStateMapper", "AgentBlenderWorldStateBackend",
    "AgentBlenderTransactionBackend",
    "AgentBlenderRuntimeBridge",
    "AgentBlenderAtomicBlendBackend",
]
__version__ = "0.1.0"
