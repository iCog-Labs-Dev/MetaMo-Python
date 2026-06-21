"""
All pygame drawing: grids, bars, panels, overlays


draw_grid(surf, offset_x, offset_y, env_state, agent_surf, mineral_surf,
          is_metamo, mot_state, alpha_dict)

draw_panel(surf, ox, oy, panel_h, label, color,
           env_state, episode, completed_episodes, total_reward,
           ep_log, summary,
           font_title, font_xs,
           mot_state, alpha_dict)
"""

import math
import pygame

from environment.gridworld import GRID_SIZE
from metamo.core import (
    arousal          as mot_arousal,
    in_safe_region   as mot_in_safe_region,
    safety_threshold as mot_safety_threshold,
)
from core.config import G_IND, G_TRANS

# Shared colour palette
BG            = ( 18,  20,  30)
GRID_LINE     = ( 40,  44,  60)
SAFE_CELL     = ( 28,  32,  46)
LAVA_COLOR    = (210,  60,  20)
MINERAL_COLOR = ( 80, 220, 140)
AGENT_BASE    = (100, 160, 255)
AGENT_META    = (255, 210,  60)
PANEL_BG      = ( 24,  26,  40)
TEXT_COLOR    = (220, 225, 240)
DIM_TEXT      = (120, 128, 160)
ACCENT_BL     = ( 80, 160, 255)
ACCENT_ME     = (255, 200,  50)
RED           = (220,  60,  60)
GREEN         = ( 80, 200, 120)
AMBER         = (255, 180,  60)
BAR_BG        = ( 50,  54,  72)

# Layout constants 
CELL     = 48
GRID_PX  = CELL * GRID_SIZE
PANEL_W  = 320



def draw_bar(surf, x, y, w, value, color, label, font_xs):
    value = max(0.0, min(1.0, float(value)))
    txt   = font_xs.render(f"{label} {value:.2f}", True, TEXT_COLOR)
    surf.blit(txt, (x, y))
    by = y + 12
    pygame.draw.rect(surf, BAR_BG, (x, by, w, 7))
    filled = int(w * value)
    if filled > 0:
        pygame.draw.rect(surf, color, (x, by, filled, 7))
    pygame.draw.rect(surf, DIM_TEXT, (x, by, w, 7), 1)


def draw_metric_line(surf, x, y, w, label, value, font_xs, value_color=TEXT_COLOR):
    lab = font_xs.render(label, True, DIM_TEXT)
    val = font_xs.render(value, True, value_color)
    surf.blit(lab, (x, y))
    surf.blit(val, (x + w - val.get_width(), y))


def draw_grid(surf, offset_x, offset_y, env_state, agent_surf, mineral_surf,
              is_metamo: bool, mot_state=None, alpha_dict=None):
    """Draw one GRID_SIZE×GRID_SIZE grid with lava, mineral, and agent."""

    lava_cells = tuple(env_state.get("lava_cells", ()))

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rx   = offset_x + c * CELL
            ry   = offset_y + r * CELL
            rect = pygame.Rect(rx, ry, CELL, CELL)

            if (r, c) in lava_cells:
                t    = pygame.time.get_ticks() / 1000.0
                wave = int(10 * math.sin(t * 2 + c * 0.7 + r * 0.4))
                col  = (
                    min(255, LAVA_COLOR[0] + wave),
                    max(0,   LAVA_COLOR[1] + wave // 2),
                    LAVA_COLOR[2],
                )
                pygame.draw.rect(surf, col, rect)
            else:
                pygame.draw.rect(surf, SAFE_CELL, rect)

            pygame.draw.rect(surf, GRID_LINE, rect, 1)

    #  Mineral with pulse glow
    mr, mc = env_state["mineral_pos"]
    mx = offset_x + mc * CELL
    my = offset_y + mr * CELL
    t       = pygame.time.get_ticks() / 1000.0
    glow_r  = int(CELL * 0.4 + 4 * math.sin(t * 3))
    glow_s  = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
    pygame.draw.circle(glow_s, (*MINERAL_COLOR, 60), (glow_r, glow_r), glow_r)
    surf.blit(glow_s, (mx + CELL // 2 - glow_r, my + CELL // 2 - glow_r))

    if mineral_surf:
        ms = pygame.transform.scale(mineral_surf, (CELL - 6, CELL - 6))
        surf.blit(ms, (mx + 3, my + 3))
    else:
        pygame.draw.circle(surf, MINERAL_COLOR,
                           (mx + CELL // 2, my + CELL // 2), CELL // 3)

    #  Agent
    ar, ac = env_state["pos"]
    ax     = offset_x + ac * CELL
    ay     = offset_y + ar * CELL
    color  = AGENT_META if is_metamo else AGENT_BASE

    if agent_surf:
        ags    = pygame.transform.scale(agent_surf, (CELL - 4, CELL - 4))
        tinted = ags.copy()
        tint   = pygame.Surface(tinted.get_size(), pygame.SRCALPHA)
        tint.fill((*color, 80))
        tinted.blit(tint, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surf.blit(ags, (ax + 2, ay + 2))
    else:
        pygame.draw.circle(surf, color,
                           (ax + CELL // 2, ay + CELL // 2), CELL // 2 - 4)

    #  Border
    border_col = ACCENT_ME if is_metamo else ACCENT_BL
    pygame.draw.rect(surf, border_col,
                     pygame.Rect(offset_x - 2, offset_y - 2,
                                 GRID_PX + 4, GRID_PX + 4), 2)


# Dashboard panel  

def draw_panel(surf, ox, oy, panel_h, label, color,
               env_state, episode, completed_episodes, total_reward,
               ep_log, summary,
               font_title, font_xs,
               mot_state=None, alpha_dict=None,
               eval_episodes=50, max_steps_ep=200):
    """Stacked dashboard panel for one agent."""

    pygame.draw.rect(surf, PANEL_BG, (ox, oy, PANEL_W, panel_h), border_radius=6)
    pygame.draw.rect(surf, color,    (ox, oy, PANEL_W, 4),
                     border_top_left_radius=6, border_top_right_radius=6)

    t = font_title.render(label, True, color)
    surf.blit(t, (ox + 12, oy + 10))

    # Live rates
    live_steps      = max(ep_log.total_steps, env_state["step"])
    live_lava_rate  = ep_log.lava_steps / live_steps if live_steps else 0.0
    live_unsafe_rate = ep_log.unsafe_rate()
    live_srv_rate    = ep_log.srv_rate()
    region_text, region_color = _environment_region(env_state)

    # Historical averages
    avg_lava   = _mean(summary, "lava_rate")
    avg_unsafe = _mean(summary, "unsafe_rate")
    avg_srv    = _mean(summary, "srv_rate")

    x = ox + 12
    w = PANEL_W - 24
    y = oy + 40
    H = 16  # line height

    rows = [
        ("Episode",       f"{episode}/{eval_episodes}",                TEXT_COLOR),
        ("Completed",     f"{completed_episodes}/{eval_episodes}",     TEXT_COLOR),
        ("Step",          f"{env_state['step']}/{max_steps_ep}",       TEXT_COLOR),
        ("Minerals",      f"{ep_log.minerals_collected}/{ep_log.minerals_spawned}", MINERAL_COLOR),
        ("Reward",        f"{total_reward:+.0f}",                      TEXT_COLOR),
        ("Lava ep/avg",   f"{live_lava_rate:.2f}/{avg_lava:.2f}",
                          RED  if live_lava_rate   else TEXT_COLOR),
        ("Unsafe ep/avg", f"{live_unsafe_rate:.2f}/{avg_unsafe:.2f}",
                          AMBER if live_unsafe_rate else TEXT_COLOR),
        ("Env region",    region_text, region_color),
    ]

    for row_label, row_value, row_color in rows:
        draw_metric_line(surf, x, y, w, row_label, row_value, font_xs, row_color)
        y += H

    if mot_state is not None:
        _draw_metamo_section(surf, x, y, w, font_xs,
                             mot_state, alpha_dict or {},
                             live_srv_rate, avg_srv)
    else:
        _draw_baseline_section(surf, x, y, w, font_xs,
                               env_state, live_srv_rate, avg_srv)


# Private helpers  

def _environment_region(env_state: dict):
    if env_state["in_lava"]:
        return "LAVA",   RED
    if env_state["lava_distance"] <= _danger_distance():
        return "DANGER", AMBER
    return "SAFE", GREEN


def _danger_distance():
    try:
        from environment.gridworld import MINERAL_SPAWN_BAND
        return MINERAL_SPAWN_BAND
    except Exception:
        return 2


def _mean(summary: dict, key: str) -> float:
    if not summary:
        return 0.0
    return float(summary.get(key, {}).get("mean", 0.0))


def _draw_metamo_section(surf, x, y, w, font_xs,
                         mot_state, alpha_dict,
                         live_srv_rate, avg_srv):
    safe       = mot_in_safe_region(mot_state)
    safe_color = GREEN if safe else RED

    draw_metric_line(surf, x, y, w,
                     "MetaMo SRV ep/avg",
                     f"{live_srv_rate:.2f}/{avg_srv:.2f}",
                     font_xs, safe_color)
    y += 16

    risk = alpha_dict.get("risk", 0.0)
    draw_metric_line(surf, x, y, w,
                     "Appraisal risk", f"{risk:.2f}",
                     font_xs, RED if risk > 0.5 else TEXT_COLOR)
    y += 20

    target_ind   = alpha_dict.get("target_individuation",  mot_state.G[G_IND])
    target_trans = alpha_dict.get("target_transcendence",  mot_state.G[G_TRANS])

    draw_bar(surf, x, y, w, mot_state.G[G_IND],   GREEN,           "Individuation",   font_xs); y += 22
    draw_bar(surf, x, y, w, target_ind,            (120, 220, 150), "Consensus Ind",   font_xs); y += 22
    draw_bar(surf, x, y, w, mot_state.G[G_TRANS],  ACCENT_ME,       "Transcendence",   font_xs); y += 22
    draw_bar(surf, x, y, w, target_trans,          (255, 225, 100), "Consensus Trans", font_xs); y += 22
    draw_bar(surf, x, y, w, mot_safety_threshold(mot_state), (60, 200, 120), "Safety threshold", font_xs); y += 22
    draw_bar(surf, x, y, w, mot_arousal(mot_state), (220, 80, 80),  "Arousal",         font_xs)


def _draw_baseline_section(surf, x, y, w, font_xs,
                            env_state, live_srv_rate, avg_srv):
    draw_metric_line(surf, x, y, w,
                     "Env SRV ep/avg",
                     f"{live_srv_rate:.2f}/{avg_srv:.2f}",
                     font_xs, AMBER if live_srv_rate else TEXT_COLOR)
    y += 22

    energy = env_state["energy"]
    ec     = GREEN if energy > 50 else AMBER if energy > 25 else RED
    draw_bar(surf, x, y, w, energy / 100.0, ec,
             f"Energy {energy:.0f}/100", font_xs)
