import numpy as np

from applications.research_assistant.schema import RESEARCH_ASSISTANT_SCHEMA
from core.schema import MotivationSchema


def _goal_idx(schema: MotivationSchema, name: str) -> int:
    return schema.goal_index(name)


def _mod_idx(schema: MotivationSchema, name: str) -> int:
    return schema.modulator_index(name)


def curiosity_to_ethics_goal_translation(
    schema: MotivationSchema = RESEARCH_ASSISTANT_SCHEMA,
) -> np.ndarray:
    """
    Map a curiosity-weighted motivational frame into an ethics-weighted frame.
    """
    g_ind = _goal_idx(schema, "individuation")
    g_trans = _goal_idx(schema, "transcendence")
    g_help = _goal_idx(schema, "help")
    g_curio = _goal_idx(schema, "curiosity")
    g_novel = _goal_idx(schema, "novelty")
    g_self = _goal_idx(schema, "self_improvement")
    g_ethic = _goal_idx(schema, "ethics")

    translation = np.eye(schema.num_goals)
    translation[g_ind] = 0.0
    translation[g_ind, g_ind] = 0.75
    translation[g_ind, g_ethic] = 0.25
    translation[g_trans, g_trans] = 0.60
    translation[g_curio, g_curio] = 0.50
    translation[g_novel, g_novel] = 0.50
    translation[g_self, g_self] = 0.70
    translation[g_self, g_ind] = 0.20
    translation[g_ethic] = 0.0
    translation[g_ethic, g_ethic] = 0.50
    translation[g_ethic, g_ind] = 0.30
    translation[g_ethic, g_help] = 0.20
    return translation


def curiosity_to_ethics_modulator_translation(
    schema: MotivationSchema = RESEARCH_ASSISTANT_SCHEMA,
) -> np.ndarray:
    """
    Translate exploratory affect into the ethics subsystem's caution language.
    """
    m_arousal = _mod_idx(schema, "arousal")
    m_approach = _mod_idx(schema, "approach")
    m_resolution = _mod_idx(schema, "resolution")
    m_valence = _mod_idx(schema, "valence")
    m_threshold = _mod_idx(schema, "threshold")
    m_securing = _mod_idx(schema, "securing")

    translation = np.eye(schema.num_modulators)
    translation[m_arousal, m_arousal] = 0.60
    translation[m_approach, m_approach] = 0.60
    translation[m_resolution, m_resolution] = 1.0
    translation[m_valence, m_valence] = 1.0
    translation[m_threshold] = 0.0
    translation[m_threshold, m_threshold] = 0.70
    translation[m_threshold, m_securing] = 0.30
    translation[m_securing] = 0.0
    translation[m_securing, m_securing] = 0.70
    translation[m_securing, m_threshold] = 0.30
    return translation
