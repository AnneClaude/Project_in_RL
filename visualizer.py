from numpy import random
import pygame
import sys
import numpy as np
from environment import TrafficEnv
from agent import QLearningAgent, SARSAAgent
from main import train_agent

# ---------------------------------------------------------------------------
# Layout Constants
# ---------------------------------------------------------------------------
SIM_WIDTH = 800          # Width of the traffic-sim area
PANEL_WIDTH = 320        # Width of the Q-value explainability panel
WIDTH = SIM_WIDTH + PANEL_WIDTH
HEIGHT = 620
RENDER_FPS = 60
LOGIC_DELAY_MS = 600

# ---------------------------------------------------------------------------
# Sleek Cyberpunk / Dark-Mode Color Palette
# ---------------------------------------------------------------------------
GRASS_GREEN   = (22,  34,  28)
ROAD_GRAY     = (38,  40,  46)
LINE_YELLOW   = (235, 165,  35)
LINE_WHITE    = (225, 225, 230)
PED_STRIPE    = (100, 104, 115)
RED_LIGHT     = (239,  83,  80)
GREEN_LIGHT   = (102, 187, 106)
YELLOW_LIGHT  = (255, 202,  40)
BULB_OFF      = ( 45,  45,  45)
HOUSING_BLACK = ( 25,  25,  25)
CAR_NS_COLOR  = ( 30, 136, 229)
CAR_EW_COLOR  = (229,  57,  53)
GLASS_COLOR   = (128, 222, 234)
TEXT_WHITE    = (245, 245, 245)

# Panel-specific palette
PANEL_BG      = ( 14,  16,  22)
PANEL_BORDER  = ( 50,  55,  70)
ACCENT_CYAN   = (  0, 229, 255)
ACCENT_PURPLE = (179,  86, 255)
ACCENT_GREEN  = (  0, 230, 118)
ACCENT_RED    = (255,  69,  58)
ACCENT_GRAY   = (120, 130, 150)
HEAT_LOW      = ( 30,  40,  80)   # low Q-value heat colour (dark blue)
HEAT_HIGH     = (255, 200,   0)   # high Q-value heat colour (amber)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def lerp_color(c1, c2, t):
    """Linearly interpolate between two RGB colours."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def draw_rounded_rect(surface, color, rect, radius=8, alpha=255):
    """Draw a filled rounded rectangle, optionally with transparency."""
    surf = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(surf, (*color, alpha), (0, 0, rect[2], rect[3]), border_radius=radius)
    surface.blit(surf, (rect[0], rect[1]))


def draw_bar(surface, x, y, w, h, value, max_value, color, bg_color=(30, 35, 50)):
    """Draw a horizontal progress bar."""
    pygame.draw.rect(surface, bg_color, (x, y, w, h), border_radius=4)
    fill = int(w * max(0.0, min(1.0, value / max(abs(max_value), 1e-6))))
    if fill > 0:
        pygame.draw.rect(surface, color, (x, y, fill, h), border_radius=4)


# ---------------------------------------------------------------------------
# Car drawing helpers (unchanged)
# ---------------------------------------------------------------------------

def draw_sleek_car(screen, x, y, is_ns, base_color):
    h_var = int((x + y) % 15) - 7
    color = (
        max(0, min(255, base_color[0] + h_var * 3)),
        max(0, min(255, base_color[1] + h_var * 3)),
        max(0, min(255, base_color[2] + h_var * 3)),
    )
    if is_ns:
        w, h = 26, 30
        pygame.draw.rect(screen, color, (x, y, w, h), border_radius=5)
        pygame.draw.rect(screen, GLASS_COLOR, (x + 3, y + 8, w - 6, 6), border_radius=1)
        pygame.draw.rect(screen, GLASS_COLOR, (x + 3, y + h - 6, w - 6, 3), border_radius=1)
        pygame.draw.circle(screen, (255, 245, 180), (x + 5, y + h - 2), 3)
        pygame.draw.circle(screen, (255, 245, 180), (x + w - 5, y + h - 2), 3)
        pygame.draw.rect(screen, (220, 20, 20), (x + 3, y + 1, 4, 2))
        pygame.draw.rect(screen, (220, 20, 20), (x + w - 7, y + 1, 4, 2))
    else:
        w, h = 30, 26
        pygame.draw.rect(screen, color, (x, y, w, h), border_radius=5)
        pygame.draw.rect(screen, GLASS_COLOR, (x + w - 14, y + 3, 6, h - 6), border_radius=1)
        pygame.draw.rect(screen, GLASS_COLOR, (x + 3, y + 3, 3, h - 6), border_radius=1)
        pygame.draw.circle(screen, (255, 245, 180), (x + w - 2, y + 5), 3)
        pygame.draw.circle(screen, (255, 245, 180), (x + w - 2, y + h - 5), 3)
        pygame.draw.rect(screen, (220, 20, 20), (x + 1, y + 3, 2, 4))
        pygame.draw.rect(screen, (220, 20, 20), (x + 1, y + h - 7, 2, 4))


class AnimatedCar:
    """Handles smooth movement of cars crossing the intersection."""
    def __init__(self, x, y, dx, dy, color, is_ns):
        self.x, self.y = x, y
        self.dx, self.dy = dx, dy
        self.color = color
        self.is_ns = is_ns

    def move(self):
        self.x += self.dx
        self.y += self.dy

    def draw(self, screen):
        draw_sleek_car(screen, self.x, self.y, self.is_ns, self.color)


# ---------------------------------------------------------------------------
# Road / intersection drawing helpers (unchanged)
# ---------------------------------------------------------------------------

def draw_base_roads(screen):
    """Draws ground layer: grass, roads, lane dividers, stop lines, crosswalks."""
    pygame.draw.rect(screen, GRASS_GREEN, (0, 0, SIM_WIDTH, HEIGHT))

    cx, cy = SIM_WIDTH // 2, HEIGHT // 2
    pygame.draw.rect(screen, ROAD_GRAY, (cx - 50, 0, 100, HEIGHT))
    pygame.draw.rect(screen, ROAD_GRAY, (0, cy - 50, SIM_WIDTH, 100))

    # Crosswalks
    for x in range(cx - 45, cx + 45, 15):
        pygame.draw.rect(screen, PED_STRIPE, (x, cy - 70, 8, 16), border_radius=1)
        pygame.draw.rect(screen, PED_STRIPE, (x, cy + 54, 8, 16), border_radius=1)
    for y in range(cy - 45, cy + 45, 15):
        pygame.draw.rect(screen, PED_STRIPE, (cx - 70, y, 16, 8), border_radius=1)
        pygame.draw.rect(screen, PED_STRIPE, (cx + 54, y, 16, 8), border_radius=1)

    # Stop lines
    pygame.draw.rect(screen, LINE_WHITE, (cx,      cy - 53, 50, 3))
    pygame.draw.rect(screen, LINE_WHITE, (cx - 50, cy + 50, 50, 3))
    pygame.draw.rect(screen, LINE_WHITE, (cx - 53, cy,      3, 50))
    pygame.draw.rect(screen, LINE_WHITE, (cx + 50, cy - 50, 3, 50))

    # Lane dividers
    for y in range(0, HEIGHT, 30):
        if y < cy - 50 or y > cy + 50:
            pygame.draw.rect(screen, LINE_YELLOW, (cx - 1, y + 5, 2, 15))
    for x in range(0, SIM_WIDTH, 30):
        if x < cx - 50 or x > cx + 50:
            pygame.draw.rect(screen, LINE_YELLOW, (x + 5, cy - 1, 15, 2))


def draw_light_bulb(screen, center, color, is_on):
    if is_on:
        glow_surf = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, 45),  (16, 16), 14)
        pygame.draw.circle(glow_surf, (*color, 100), (16, 16),  8)
        screen.blit(glow_surf, (center[0] - 16, center[1] - 16))
        pygame.draw.circle(screen, color, center, 4)
        pygame.draw.circle(screen, (255, 255, 255), (int(center[0] - 1.5), int(center[1] - 1.5)), 1)
    else:
        pygame.draw.circle(screen, BULB_OFF, center, 4)


def draw_overhead_lights(screen, light_state):
    ns_red    = (light_state in (1, 3))
    ns_yellow = (light_state == 2)
    ns_green  = (light_state == 0)
    ew_red    = (light_state in (0, 2))
    ew_yellow = (light_state == 3)
    ew_green  = (light_state == 1)

    cx, cy = SIM_WIDTH // 2, HEIGHT // 2

    pygame.draw.rect(screen, HOUSING_BLACK, (cx + 5,  cy - 70,  54, 16), border_radius=4)
    draw_light_bulb(screen, (cx + 14, cy - 62), RED_LIGHT,    ns_red)
    draw_light_bulb(screen, (cx + 32, cy - 62), YELLOW_LIGHT, ns_yellow)
    draw_light_bulb(screen, (cx + 50, cy - 62), GREEN_LIGHT,  ns_green)

    pygame.draw.rect(screen, HOUSING_BLACK, (cx - 59, cy + 55,  54, 16), border_radius=4)
    draw_light_bulb(screen, (cx - 50, cy + 63), RED_LIGHT,    ns_red)
    draw_light_bulb(screen, (cx - 32, cy + 63), YELLOW_LIGHT, ns_yellow)
    draw_light_bulb(screen, (cx - 14, cy + 63), GREEN_LIGHT,  ns_green)

    pygame.draw.rect(screen, HOUSING_BLACK, (cx - 70, cy + 5,  16, 54), border_radius=4)
    draw_light_bulb(screen, (cx - 62, cy + 14), RED_LIGHT,    ew_red)
    draw_light_bulb(screen, (cx - 62, cy + 32), YELLOW_LIGHT, ew_yellow)
    draw_light_bulb(screen, (cx - 62, cy + 50), GREEN_LIGHT,  ew_green)

    pygame.draw.rect(screen, HOUSING_BLACK, (cx + 55, cy - 59,  16, 54), border_radius=4)
    draw_light_bulb(screen, (cx + 63, cy - 50), RED_LIGHT,    ew_red)
    draw_light_bulb(screen, (cx + 63, cy - 32), YELLOW_LIGHT, ew_yellow)
    draw_light_bulb(screen, (cx + 63, cy - 14), GREEN_LIGHT,  ew_green)


def draw_queued_cars(screen, ns_queue, ew_queue):
    cx, cy = SIM_WIDTH // 2, HEIGHT // 2
    for i in range(ns_queue):
        draw_sleek_car(screen, cx + 12, cy - 85 - i * 38, True, CAR_NS_COLOR)
    for i in range(ew_queue):
        draw_sleek_car(screen, cx - 85 - i * 38, cy + 12, False, CAR_EW_COLOR)


def draw_hud(screen, font, ns, ew, agent_action, mode_name, phase_name, rates):
    hud_surface = pygame.Surface((340, 130), pygame.SRCALPHA)
    hud_surface.fill((20, 20, 20, 220))
    screen.blit(hud_surface, (20, 20))
    pygame.draw.rect(screen, LINE_WHITE, (20, 20, 340, 130), width=1, border_radius=4)

    title_font = pygame.font.SysFont(None, 22, bold=True)
    screen.blit(title_font.render(f"MODE: {mode_name}", True, LINE_YELLOW), (35, 30))
    screen.blit(font.render(f"PHASE: {phase_name} (NS: {int(rates[0]*100)}%, EW: {int(rates[1]*100)}%)", True, LINE_WHITE), (35, 55))
    screen.blit(font.render(f"NS Lane Queue: {ns}/5  |  EW Lane Queue: {ew}/5", True, TEXT_WHITE), (35, 80))
    screen.blit(font.render(f"Decision: {agent_action}", True, TEXT_WHITE), (35, 105))


# ---------------------------------------------------------------------------
# *** NEW: Live Q-Value Explainability Panel ***
# ---------------------------------------------------------------------------

def get_q_values_for_state(agent, state, mode):
    """
    Returns (q_keep, q_switch) for the current state and agent type.
    Returns None if the agent type does not support Q-value extraction.
    """
    if agent is None or mode not in ("q_learning", "sarsa", "dqn"):
        return None

    ns, ew, light = state

    if mode in ("q_learning", "sarsa"):
        # Tabular: direct lookup
        # Map internal 4-state light to 2-state (yellow phases -> collapse to parent)
        q_light = min(light, 1)   # 0 or 1 for the q-table (which only has 0..1)
        q_vals = agent.q_table[ns, ew, q_light]
        return float(q_vals[0]), float(q_vals[1])

    elif mode == "dqn":
        import torch
        state_vec = agent.preprocess_state(state)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(agent.device)
        agent.q_net.eval()
        with torch.no_grad():
            q_out = agent.q_net(state_tensor)
        vals = q_out.squeeze().cpu().numpy()
        return float(vals[0]), float(vals[1])

    return None


def get_full_q_slice(agent, light_state, mode, max_queue=5):
    """
    Returns a (max_queue+1 x max_queue+1) numpy array of the best-action Q-value
    (or the advantage = Q_switch - Q_keep) for every (ns, ew) combo at the given
    light phase. Used to build the heatmap.
    Returns None for DQN (too expensive to iterate all states visually).
    """
    if agent is None or mode not in ("q_learning", "sarsa"):
        return None

    q_light = min(light_state, 1)
    size = max_queue + 1
    # We'll show: max(Q_keep, Q_switch) — the "value" of that state
    grid = np.zeros((size, size), dtype=np.float32)
    for ns in range(size):
        for ew in range(size):
            q = agent.q_table[ns, ew, q_light]
            grid[ns, ew] = float(np.max(q))
    return grid


def draw_q_panel(screen, fonts, agent, state, mode, max_queue=5):
    """
    Draws the full live Q-value explainability panel on the right side.

    Sections:
      1. Panel header
      2. Current-state Q-value bar chart (Keep vs Switch)
      3. State info badges
      4. Q-table heatmap (tabular agents only)
    """
    px = SIM_WIDTH          # Panel x-start
    py = 0
    pw = PANEL_WIDTH
    ph = HEIGHT

    font_sm  = fonts["sm"]
    font_md  = fonts["md"]
    font_lg  = fonts["lg"]
    font_xs  = fonts["xs"]

    ns, ew, light = state
    q_light = min(light, 1)

    # ---- Background ----
    pygame.draw.rect(screen, PANEL_BG, (px, py, pw, ph))
    pygame.draw.line(screen, PANEL_BORDER, (px, 0), (px, ph), 2)

    # ---- Header ----
    header_h = 50
    draw_rounded_rect(screen, (20, 24, 40), (px, py, pw, header_h), radius=0, alpha=255)
    pygame.draw.line(screen, PANEL_BORDER, (px, py + header_h), (px + pw, py + header_h), 1)

    title_surf = font_lg.render("⚡ Agent Intelligence", True, ACCENT_CYAN)
    screen.blit(title_surf, (px + 12, py + 8))
    sub_surf = font_xs.render("Live Q-Value Visualization", True, ACCENT_GRAY)
    screen.blit(sub_surf, (px + 14, py + 32))

    y = py + header_h + 10

    # ---- State badges ----
    label_color = ACCENT_GRAY
    screen.blit(font_sm.render("CURRENT STATE", True, label_color), (px + 12, y))
    y += 20

    # NS queue badge
    ns_col = lerp_color(ACCENT_GREEN, ACCENT_RED, ns / max_queue)
    draw_rounded_rect(screen, ns_col, (px + 10, y, 88, 28), radius=6, alpha=200)
    screen.blit(font_sm.render(f"NS Queue: {ns}", True, TEXT_WHITE), (px + 16, y + 7))

    # EW queue badge
    ew_col = lerp_color(ACCENT_GREEN, ACCENT_RED, ew / max_queue)
    draw_rounded_rect(screen, ew_col, (px + 108, y, 88, 28), radius=6, alpha=200)
    screen.blit(font_sm.render(f"EW Queue: {ew}", True, TEXT_WHITE), (px + 114, y + 7))

    # Light phase badge
    light_names = {0: "NS Green", 1: "EW Green", 2: "NS→EW (Yel)", 3: "EW→NS (Yel)"}
    light_cols  = {0: GREEN_LIGHT, 1: GREEN_LIGHT, 2: YELLOW_LIGHT, 3: YELLOW_LIGHT}
    draw_rounded_rect(screen, light_cols.get(light, ACCENT_GRAY), (px + 206, y, 100, 28), radius=6, alpha=190)
    screen.blit(font_xs.render(light_names.get(light, "?"), True, HOUSING_BLACK), (px + 212, y + 8))
    y += 42

    # ---- Q-Value bar chart ----
    pygame.draw.line(screen, PANEL_BORDER, (px + 10, y), (px + pw - 10, y), 1)
    y += 10
    screen.blit(font_sm.render("Q-VALUES FOR THIS STATE", True, label_color), (px + 12, y))
    y += 20

    q_vals = get_q_values_for_state(agent, state, mode)

    if q_vals is not None:
        q_keep, q_switch = q_vals
        bar_max = max(abs(q_keep), abs(q_switch), 1.0)

        # Decide which is preferred
        preferred = "KEEP" if q_keep >= q_switch else "SWITCH"

        for label, val, col, badge in [
            ("KEEP LIGHT",   q_keep,   ACCENT_CYAN,   preferred == "KEEP"),
            ("SWITCH LIGHT", q_switch, ACCENT_PURPLE, preferred == "SWITCH"),
        ]:
            # Label row
            lbl_surf = font_sm.render(label, True, col)
            screen.blit(lbl_surf, (px + 12, y))
            val_surf = font_sm.render(f"{val:+.3f}", True, TEXT_WHITE)
            screen.blit(val_surf, (px + pw - val_surf.get_width() - 14, y))
            y += 18

            # Bar
            bar_w = pw - 24
            norm = abs(val) / bar_max
            fill = int(bar_w * norm)
            pygame.draw.rect(screen, (30, 35, 55), (px + 12, y, bar_w, 16), border_radius=5)
            if fill > 0:
                bar_col = col if val >= 0 else ACCENT_RED
                pygame.draw.rect(screen, bar_col, (px + 12, y, fill, 16), border_radius=5)
                # Shimmer highlight
                pygame.draw.rect(screen, (*bar_col[:3], 80),
                                  (px + 12, y, fill, 6), border_radius=5)

            # "★ PREFERRED" badge
            if badge:
                star_surf = font_xs.render("★ PREFERRED", True, LINE_YELLOW)
                screen.blit(star_surf, (px + pw - star_surf.get_width() - 14, y + 1))
            y += 22

        y += 4

        # Confidence strip
        advantage = q_switch - q_keep
        conf_pct = abs(advantage) / (bar_max + 1e-6)
        conf_str = f"Confidence: {int(conf_pct * 100)}%  |  Δ = {advantage:+.3f}"
        screen.blit(font_xs.render(conf_str, True, ACCENT_GRAY), (px + 12, y))
        y += 20

        # TD explanation hint
        hint = "Agent strongly prefers switching" if advantage > 1.0 else \
               "Agent slightly prefers switching" if advantage > 0.1 else \
               "Agent strongly prefers keeping" if advantage < -1.0 else \
               "Agent slightly prefers keeping" if advantage < -0.1 else \
               "Agent is indifferent (learning...)"
        hint_surf = font_xs.render(hint, True, ACCENT_GRAY)
        screen.blit(hint_surf, (px + 12, y))
        y += 18

    else:
        screen.blit(font_sm.render("N/A (baseline mode)", True, ACCENT_GRAY), (px + 14, y))
        y += 40

    # ---- Q-Table Heatmap (tabular agents only) ----
    pygame.draw.line(screen, PANEL_BORDER, (px + 10, y), (px + pw - 10, y), 1)
    y += 10

    if mode in ("q_learning", "sarsa") and agent is not None:
        screen.blit(font_sm.render("Q-TABLE HEATMAP", True, label_color), (px + 12, y))
        phase_label = "NS Green" if q_light == 0 else "EW Green"
        screen.blit(font_xs.render(f"Phase: {phase_label}  |  colour = max Q(s)", True, ACCENT_GRAY), (px + 12, y + 16))
        y += 36

        grid = get_full_q_slice(agent, light, mode, max_queue)
        if grid is not None:
            size = max_queue + 1        # 6
            cell = (pw - 40) // size    # ~46 px per cell
            g_min, g_max = grid.min(), grid.max()
            g_range = max(g_max - g_min, 1e-6)

            # Axis labels
            screen.blit(font_xs.render("NS →", True, ACCENT_GRAY), (px + 12, y - 14))
            screen.blit(font_xs.render("EW ↓", True, ACCENT_GRAY), (px + 12, y))

            for ns_i in range(size):
                for ew_i in range(size):
                    t = (grid[ns_i, ew_i] - g_min) / g_range
                    cell_col = lerp_color(HEAT_LOW, HEAT_HIGH, t)
                    cx_ = px + 12 + ns_i * cell
                    cy_ = y + 16 + ew_i * cell

                    pygame.draw.rect(screen, cell_col, (cx_, cy_, cell - 2, cell - 2), border_radius=3)

                    # Highlight current state cell
                    if ns_i == ns and ew_i == ew:
                        pygame.draw.rect(screen, TEXT_WHITE, (cx_, cy_, cell - 2, cell - 2),
                                         width=2, border_radius=3)

            # Colour-bar legend (vertical, right of heatmap)
            legend_x = px + 12 + size * cell + 6
            legend_h = size * cell
            for row in range(legend_h):
                t_row = 1.0 - row / max(legend_h - 1, 1)
                c = lerp_color(HEAT_LOW, HEAT_HIGH, t_row)
                pygame.draw.line(screen, c, (legend_x, y + 16 + row), (legend_x + 12, y + 16 + row))
            screen.blit(font_xs.render("Hi", True, TEXT_WHITE), (legend_x + 14, y + 16))
            screen.blit(font_xs.render("Lo", True, TEXT_WHITE), (legend_x + 14, y + 16 + legend_h - 12))

            y += 16 + size * cell + 8

    elif mode == "dqn" and agent is not None:
        screen.blit(font_sm.render("Q-TABLE HEATMAP", True, label_color), (px + 12, y))
        y += 18
        note1 = font_xs.render("DQN uses a neural network.", True, ACCENT_GRAY)
        note2 = font_xs.render("No fixed table to visualise.", True, ACCENT_GRAY)
        note3 = font_xs.render("See bar chart above for live", True, ACCENT_GRAY)
        note4 = font_xs.render("inference output.", True, ACCENT_GRAY)
        for surf in (note1, note2, note3, note4):
            screen.blit(surf, (px + 14, y))
            y += 15
    else:
        screen.blit(font_sm.render("Q-TABLE HEATMAP", True, label_color), (px + 12, y))
        y += 18
        screen.blit(font_xs.render("Not available in baseline mode.", True, ACCENT_GRAY), (px + 14, y))

    # ---- Footer / legend ----
    footer_y = ph - 36
    pygame.draw.line(screen, PANEL_BORDER, (px + 10, footer_y), (px + pw - 10, footer_y), 1)
    screen.blit(font_xs.render("■ CYAN = Keep  ■ PURPLE = Switch  ■ White = Current state",
                                True, ACCENT_GRAY), (px + 12, footer_y + 8))
    screen.blit(font_xs.render("Q-Learning XAI  |  Live Bellman Values",
                                True, (60, 65, 80)), (px + 12, footer_y + 20))


# ---------------------------------------------------------------------------
# Main visualisation loop
# ---------------------------------------------------------------------------

def run_visualization(agent, env, mode):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL Traffic Signal Visualizer  |  Live Q-Value XAI")
    clock = pygame.time.Clock()

    # Font sizes
    fonts = {
        "xs": pygame.font.SysFont("Consolas", 13),
        "sm": pygame.font.SysFont("Consolas", 15, bold=True),
        "md": pygame.font.SysFont("Consolas", 17),
        "lg": pygame.font.SysFont("Consolas", 20, bold=True),
    }
    font_hud = pygame.font.SysFont(None, 20)

    state = env.reset()
    ns, ew, light = state
    agent_action = "INITIALIZING..."

    animating_cars = []
    last_logic_time = pygame.time.get_ticks()

    mode_names = {
        "q_learning": "Trained Q-Learning Agent",
        "sarsa":      "Trained SARSA Agent",
        "dqn":        "Trained DQN Agent",
        "lqf":        "Longest Queue First Heuristic",
        "fixed":      "Fixed-Time Switch (5-steps)",
        "random":     "Random Switch",
    }
    mode_name = mode_names.get(mode, mode.upper())

    while True:
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if current_time - last_logic_time > LOGIC_DELAY_MS:
            prev_ns, prev_ew, prev_light = state

            if prev_light in [2, 3]:
                raw_action = 0
                agent_action = "YELLOW TRANSITION"
            else:
                if mode in ("q_learning", "sarsa", "dqn") and agent is not None:
                    raw_action = agent.choose_action(state, epsilon=0.0)
                elif mode == "lqf":
                    if prev_light == 0:
                        raw_action = 1 if prev_ew > prev_ns and env.time_since_switch >= 3 else 0
                    elif prev_light == 1:
                        raw_action = 1 if prev_ns > prev_ew and env.time_since_switch >= 3 else 0
                    else:
                        raw_action = 0
                elif mode == "fixed":
                    raw_action = 1 if env.steps > 0 and env.steps % 5 == 0 else 0
                else:
                    raw_action = random.choice([0, 1])

                agent_action = "SWITCH LIGHT" if raw_action == 1 else "HOLD LIGHT"

            state, reward, done, _ = env.step(raw_action)

            if done:
                state = env.reset()

            if (prev_light in (0, 3)) and prev_ns > 0:
                animating_cars.append(
                    AnimatedCar(SIM_WIDTH // 2 + 12, HEIGHT // 2 - 85, 0, 8, CAR_NS_COLOR, True)
                )
            elif (prev_light in (1, 2)) and prev_ew > 0:
                animating_cars.append(
                    AnimatedCar(SIM_WIDTH // 2 - 85, HEIGHT // 2 + 12, 8, 0, CAR_EW_COLOR, False)
                )

            ns, ew, light = state
            last_logic_time = current_time

        for car in animating_cars:
            car.move()
        animating_cars = [c for c in animating_cars
                          if -50 < c.x < SIM_WIDTH + 50 and -50 < c.y < HEIGHT + 50]

        # ---- Render ----
        draw_base_roads(screen)
        draw_queued_cars(screen, ns, ew)
        for car in animating_cars:
            car.draw(screen)
        draw_overhead_lights(screen, light)

        phase_name = env.get_traffic_phase()
        rates = env.get_arrival_probabilities()
        draw_hud(screen, font_hud, ns, ew, agent_action, mode_name, phase_name, rates)

        # *** Draw the live Q-value panel ***
        draw_q_panel(screen, fonts, agent, state, mode, max_queue=5)

        pygame.display.flip()
        clock.tick(RENDER_FPS)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Traffic Signal Control Visualizer with Live Q-Value XAI")
    parser.add_argument(
        "--mode",
        type=str,
        default="q_learning",
        choices=["q_learning", "sarsa", "dqn", "lqf", "fixed", "random"],
        help="Control mode (default: q_learning)",
    )
    args = parser.parse_args()

    env = TrafficEnv(max_queue=5, max_steps=100)
    agent = None

    if args.mode == "q_learning":
        agent = QLearningAgent(max_queue=5)
        if os.path.exists("q_table_qlearning.npy"):
            print("Loading trained Q-table for Q-Learning...")
            agent.q_table = np.load("q_table_qlearning.npy")
        else:
            print("Q-table not found. Training Q-Learning agent (500 episodes)...")
            train_agent(env, agent, num_episodes=500)

    elif args.mode == "sarsa":
        agent = SARSAAgent(max_queue=5)
        if os.path.exists("q_table_sarsa.npy"):
            print("Loading trained Q-table for SARSA...")
            agent.q_table = np.load("q_table_sarsa.npy")
        else:
            print("Q-table not found. Training SARSA agent (500 episodes)...")
            from compare_algorithms import train_sarsa
            train_sarsa(env, agent, num_episodes=500)

    elif args.mode == "dqn":
        from dqn_agent import DQNAgent
        agent = DQNAgent(state_dim=6, action_dim=2)
        if os.path.exists("dqn_model.pth"):
            print("Loading trained weights for DQN...")
            agent.load("dqn_model.pth")
        else:
            print("DQN model not found. Training DQN agent (500 episodes)...")
            from compare_algorithms import train_dqn
            train_dqn(env, agent, num_episodes=500)

    print(f"\nLaunching visualizer in mode: {args.mode.upper()}")
    run_visualization(agent, env, args.mode)