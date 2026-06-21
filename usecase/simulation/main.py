"""
pygame event loop and top-level orchestration.

Controls
--------
SPACE       pause / resume
R           reset both agents to a fresh evaluation
F / UP      increase simulation speed
S / DOWN    decrease simulation speed
Q / ESC     quit
"""

import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
USECASE_ROOT = os.path.dirname(ROOT)
PROJECT_ROOT = os.path.dirname(USECASE_ROOT)

for path in (PROJECT_ROOT, USECASE_ROOT, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import pygame

from agents.baseline_agent import BaselineAgent
from agents.metamo_agent   import MetaMoAgent
from metrics.collector     import MetricsCollector

from runner import (
    TRAIN_EPISODES,
    EVAL_EPISODES,
    MAX_STEPS_EP,
    train_agent,
    new_episode,
    clear_episode_logs,
    step_baseline,
    step_metamo,
)
from renderer import (
    CELL, GRID_PX, PANEL_W,
    BG, DIM_TEXT, TEXT_COLOR, ACCENT_BL, ACCENT_ME,
    draw_grid,
    draw_panel,
)

# Speed settings  

DEFAULT_STEPS_PER_SECOND = 8.0
MIN_STEPS_PER_SECOND     = 1.0
MAX_STEPS_PER_SECOND     = 30.0

# Layout 

GAP         = 28
DASH_GAP    = 16
MARGIN      = 20
GRID_OY     = 70
DASH_PANEL_H = 335
FPS         = 60

WIN_W = MARGIN * 2 + GRID_PX * 2 + GAP + DASH_GAP + PANEL_W
WIN_H = GRID_OY + DASH_PANEL_H * 2 + DASH_GAP + 42

ASSET_DIR = os.path.join(USECASE_ROOT, "assets")
# Key sets  

FASTER_KEYS = {
    pygame.K_f, pygame.K_UP, pygame.K_EQUALS,
    getattr(pygame, "K_PLUS",    pygame.K_EQUALS),
    getattr(pygame, "K_KP_PLUS", pygame.K_EQUALS),
}
SLOWER_KEYS = {
    pygame.K_s, pygame.K_DOWN, pygame.K_MINUS,
    getattr(pygame, "K_KP_MINUS", pygame.K_MINUS),
}


# Entry point  

def main():
    # Training
    baseline = BaselineAgent(seed=0)
    metamo   = MetaMoAgent(seed=0)
    train_agent(baseline, "Baseline RL", TRAIN_EPISODES, seed_offset=0)
    train_agent(metamo,   "MetaMo  RL",  TRAIN_EPISODES, seed_offset=0)

    bl_metrics = MetricsCollector("Baseline")
    mm_metrics = MetricsCollector("MetaMo")

    # Pygame init
    pygame.init()
    try:
        pygame.mixer.init()
    except pygame.error:
        pass

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("MetaMo vs Baseline RL — GridWorld Evaluation")
    clock = pygame.time.Clock()

    font_title = pygame.font.Font(None, 28)
    font_xs    = pygame.font.Font(None, 17)
    font_lg    = pygame.font.Font(None, 40)

    # Assets
    try:
        agent_img = pygame.image.load(
            os.path.join(ASSET_DIR, "cat.jpg")).convert_alpha()
    except Exception:
        agent_img = None

    mineral_surf = None 

    def _play_clank():
        try:
            clank_sound.play()
        except Exception:
            pass

    try:
        clank_sound = pygame.mixer.Sound(os.path.join(ASSET_DIR, "clank.wav"))
        clank_sound.set_volume(0.4)
        clank_fn = _play_clank
    except Exception:
        clank_fn = None

    # Simulation state
    episode            = 1
    completed_episodes = 0
    env_bl, env_mm, s_bl, s_mm = new_episode(episode, baseline, metamo)
    done_bl, done_mm, reward_bl, reward_mm, lava_bl, lava_mm, ep_log_bl, ep_log_mm = \
        clear_episode_logs()
    alpha_mm = {"risk": 0, "urgency": 0, "eu": 0}

    steps_per_second    = DEFAULT_STEPS_PER_SECOND
    step_accum_ms       = 0.0
    paused              = False
    evaluation_complete = False

    # Main loop  
    running = True
    while running:
        dt_ms = clock.tick(FPS)

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                k = event.key

                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif k == pygame.K_SPACE:
                    paused = not paused

                elif k == pygame.K_r:
                    if evaluation_complete:
                        bl_metrics.episodes.clear()
                        mm_metrics.episodes.clear()
                        completed_episodes = 0
                        episode = 1
                    evaluation_complete = False
                    paused              = False
                    step_accum_ms       = 0.0
                    env_bl, env_mm, s_bl, s_mm = new_episode(episode, baseline, metamo)
                    done_bl, done_mm, reward_bl, reward_mm, lava_bl, lava_mm, \
                        ep_log_bl, ep_log_mm = clear_episode_logs()

                elif k in FASTER_KEYS:
                    steps_per_second = min(steps_per_second + 1.0, MAX_STEPS_PER_SECOND)

                elif k in SLOWER_KEYS:
                    steps_per_second = max(steps_per_second - 1.0, MIN_STEPS_PER_SECOND)

        # Simulation steps
        if not paused and not evaluation_complete:
            step_accum_ms    += dt_ms
            step_interval_ms  = 1000.0 / steps_per_second

            while step_accum_ms >= step_interval_ms and not evaluation_complete:
                step_accum_ms -= step_interval_ms

                if not done_bl:
                    s_bl, reward_bl, lava_bl, done_bl = step_baseline(
                        baseline, env_bl, s_bl, ep_log_bl,
                        reward_bl, lava_bl, clank_fn)

                if not done_mm:
                    s_mm, reward_mm, lava_mm, done_mm, alpha_mm = step_metamo(
                        metamo, env_mm, s_mm, ep_log_mm,
                        reward_mm, lava_mm, clank_fn)

            # Auto-advance when both agents finish
            if done_bl and done_mm:
                ep_log_bl.total_reward    = reward_bl
                ep_log_bl.lava_steps      = lava_bl
                ep_log_bl.total_steps     = env_bl.step_count
                ep_log_bl.minerals_spawned = env_bl.minerals_spawned

                ep_log_mm.total_reward    = reward_mm
                ep_log_mm.lava_steps      = lava_mm
                ep_log_mm.total_steps     = env_mm.step_count
                ep_log_mm.minerals_spawned = env_mm.minerals_spawned

                bl_metrics.add(ep_log_bl)
                mm_metrics.add(ep_log_mm)
                completed_episodes += 1

                if completed_episodes >= EVAL_EPISODES:
                    evaluation_complete = True
                    paused = True
                else:
                    episode = completed_episodes + 1
                    env_bl, env_mm, s_bl, s_mm = new_episode(episode, baseline, metamo)
                    done_bl, done_mm, reward_bl, reward_mm, lava_bl, lava_mm, \
                        ep_log_bl, ep_log_mm = clear_episode_logs()

                step_accum_ms = 0.0

        # Draw
        screen.fill(BG)

        title = font_lg.render(
            "MetaMo  vs  Baseline RL  —  GridWorld", True, TEXT_COLOR)
        screen.blit(title, (WIN_W // 2 - title.get_width() // 2, 8))

        BL_GRID_X  = MARGIN
        MM_GRID_X  = BL_GRID_X + GRID_PX + GAP
        DASH_X     = MM_GRID_X + GRID_PX + DASH_GAP
        BL_PANEL_Y = GRID_OY
        MM_PANEL_Y = GRID_OY + DASH_PANEL_H + DASH_GAP

        draw_grid(screen, BL_GRID_X, GRID_OY, s_bl,
                  agent_img, mineral_surf, is_metamo=False)

        draw_grid(screen, MM_GRID_X, GRID_OY, s_mm,
                  agent_img, mineral_surf, is_metamo=True,
                  mot_state=metamo.mot, alpha_dict=alpha_mm)

        # Labels above grids
        bl_lbl = font_title.render("BASELINE RL", True, ACCENT_BL)
        mm_lbl = font_title.render("MetaMo RL",   True, ACCENT_ME)
        screen.blit(bl_lbl, (BL_GRID_X + GRID_PX // 2 - bl_lbl.get_width() // 2, GRID_OY - 25))
        screen.blit(mm_lbl, (MM_GRID_X + GRID_PX // 2 - mm_lbl.get_width() // 2, GRID_OY - 25))

        bl_summary = bl_metrics.summary()
        mm_summary = mm_metrics.summary()

        draw_panel(screen, DASH_X, BL_PANEL_Y, DASH_PANEL_H,
                   "Baseline RL", ACCENT_BL,
                   s_bl, episode, completed_episodes, reward_bl,
                   ep_log_bl, bl_summary,
                   font_title, font_xs,
                   eval_episodes=EVAL_EPISODES, max_steps_ep=MAX_STEPS_EP)

        draw_panel(screen, DASH_X, MM_PANEL_Y, DASH_PANEL_H,
                   "MetaMo RL", ACCENT_ME,
                   s_mm, episode, completed_episodes, reward_mm,
                   ep_log_mm, mm_summary,
                   font_title, font_xs,
                   mot_state=metamo.mot, alpha_dict=alpha_mm,
                   eval_episodes=EVAL_EPISODES, max_steps_ep=MAX_STEPS_EP)

        # Footer
        ctrl = font_xs.render(
            "SPACE: pause   R: reset   F/UP: faster   S/DOWN: slower   Q: quit"
            f"   |   Speed: {steps_per_second:.0f} steps/s"
            f"   |   Episodes completed: {completed_episodes}/{EVAL_EPISODES}",
            True, DIM_TEXT)
        screen.blit(ctrl, (WIN_W // 2 - ctrl.get_width() // 2, WIN_H - 22))

        if evaluation_complete:
            msg = font_lg.render("EVALUATION COMPLETE", True, (255, 220, 60))
            screen.blit(msg, (WIN_W // 2 - msg.get_width() // 2, WIN_H // 2 - 20))
        elif paused:
            msg = font_lg.render("PAUSED", True, (255, 220, 60))
            screen.blit(msg, (WIN_W // 2 - msg.get_width() // 2, WIN_H // 2 - 20))

        pygame.display.flip()

    #  Final summary 
    print("\n" + "=" * 55)
    for mc in [bl_metrics, mm_metrics]:
        s = mc.summary()
        if not s:
            continue
        print(f"\n[ {s['label']} ]  ({s['n_episodes']} episodes)")
        print(f"  Completion rate : {s['completion_rate']['mean']:.3f} ± {s['completion_rate']['std']:.3f}")
        print(f"  Lava rate       : {s['lava_rate']['mean']:.3f} ± {s['lava_rate']['std']:.3f}")
        print(f"  SRV rate        : {s['srv_rate']['mean']:.3f} ± {s['srv_rate']['std']:.3f}")
        print(f"  Unsafe-zone rate: {s['unsafe_rate']['mean']:.3f} ± {s['unsafe_rate']['std']:.3f}")
        print(f"  Recovery time   : {s['recovery_time']['mean']:.1f} ± {s['recovery_time']['std']:.1f}")
        print(f"  Total reward    : {s['total_reward']['mean']:.1f} ± {s['total_reward']['std']:.1f}")
    print("=" * 55)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
