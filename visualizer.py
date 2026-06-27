"""
Traffic Signal Control — Live Visualizer
=========================================
Simulation (left):  Stylised top-down intersection with animated cars.
Dashboard (right):  3 focused sections:
    1. DECISION  — what the agent chose, in huge text
    2. REASONING — Q-score comparison (why it chose that)
    3. LIVE CHART — rolling reward-history line graph (unique feature)
"""
from numpy import random
import pygame
import sys
import math
import numpy as np
from collections import deque
from environment import TrafficEnv
from agent import QLearningAgent, SARSAAgent

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
SIM_WIDTH     = 780
PANEL_WIDTH   = 340
WIDTH         = SIM_WIDTH + PANEL_WIDTH
HEIGHT        = 660
RENDER_FPS    = 60
LOGIC_DELAY_MS = 550

# ─────────────────────────────────────────────────────────────────────────────
# Colour Palette  ── deep navy / neon accent
# ─────────────────────────────────────────────────────────────────────────────
# Simulation
BG_GRASS     = (18,  28,  22)
ASPHALT      = (40,  44,  55)
ASPHALT_DARK = (32,  36,  46)
LANE_DASH    = (210, 185,  60)   # yellow centre line
LANE_WHITE   = (200, 210, 220)   # edge / stop line
CROSS_STRIPE = ( 70,  76,  90)   # pedestrian crossing
TL_HOUSING   = ( 20,  20,  24)
RED_BULB     = (240,  70,  70)
YEL_BULB     = (255, 205,  40)
GRN_BULB     = ( 80, 210, 100)
BULB_OFF     = ( 45,  48,  58)
CAR_NS       = ( 30, 140, 230)   # blue  — north-south cars
CAR_EW       = (220,  70,  60)   # red   — east-west  cars
GLASS        = (120, 210, 230)

# Panel
P_BG         = ( 12,  14,  20)   # panel background
P_CARD       = ( 20,  24,  36)   # card background
P_BORDER     = ( 42,  48,  64)   # divider / outline
P_TEXT       = (220, 225, 235)   # body text
P_DIM        = ( 90, 100, 120)   # dim label text
C_CYAN       = (  0, 220, 255)   # hold-light accent
C_PURPLE     = (170,  80, 255)   # switch-light accent
C_GREEN      = ( 50, 210, 110)
C_RED        = (240,  70,  70)
C_YELLOW     = (255, 200,  40)
C_AMBER      = (255, 155,  40)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def lc(c1, c2, t):
    """Lerp between two colours."""
    t = max(0.0, min(1.0, float(t)))
    return (int(c1[0]+(c2[0]-c1[0])*t),
            int(c1[1]+(c2[1]-c1[1])*t),
            int(c1[2]+(c2[2]-c1[2])*t))


def pill(surf, colour, rect, r=8, alpha=255, border=0, border_col=None):
    """Draw a rounded rectangle, with optional border."""
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(s, (*colour, alpha), (0, 0, rect[2], rect[3]), border_radius=r)
    surf.blit(s, (rect[0], rect[1]))
    if border:
        pygame.draw.rect(surf, border_col or colour, rect, width=border, border_radius=r)


# ─────────────────────────────────────────────────────────────────────────────
# Car drawing
# ─────────────────────────────────────────────────────────────────────────────
def draw_car(screen, x, y, is_ns, base_color):
    h_var = int((x * 3 + y * 7) % 18) - 9
    col = tuple(max(0, min(255, c + h_var * 2)) for c in base_color)
    if is_ns:
        w, h = 24, 28
        pygame.draw.rect(screen, col,   (x, y, w, h), border_radius=5)
        pygame.draw.rect(screen, GLASS, (x+3, y+6,  w-6, 6), border_radius=2)
        pygame.draw.rect(screen, GLASS, (x+3, y+h-8, w-6, 5), border_radius=2)
        pygame.draw.circle(screen, (255, 240, 160), (x+4,   y+h-2), 3)
        pygame.draw.circle(screen, (255, 240, 160), (x+w-4, y+h-2), 3)
    else:
        w, h = 28, 24
        pygame.draw.rect(screen, col,   (x, y, w, h), border_radius=5)
        pygame.draw.rect(screen, GLASS, (x+w-12, y+3, 8, h-6), border_radius=2)
        pygame.draw.rect(screen, GLASS, (x+4,    y+3, 5, h-6), border_radius=2)
        pygame.draw.circle(screen, (255, 240, 160), (x+w-2, y+4),   3)
        pygame.draw.circle(screen, (255, 240, 160), (x+w-2, y+h-4), 3)


class AnimatedCar:
    def __init__(self, x, y, dx, dy, color, is_ns):
        self.x, self.y, self.dx, self.dy = x, y, dx, dy
        self.color, self.is_ns = color, is_ns

    def move(self):
        self.x += self.dx
        self.y += self.dy

    def draw(self, screen):
        draw_car(screen, self.x, self.y, self.is_ns, self.color)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation drawing
# ─────────────────────────────────────────────────────────────────────────────
def draw_road(screen):
    """Draw the full intersection: asphalt, crosswalks, markings."""
    pygame.draw.rect(screen, BG_GRASS, (0, 0, SIM_WIDTH, HEIGHT))

    cx, cy = SIM_WIDTH // 2, HEIGHT // 2
    ROAD_W = 110   # half-width of road

    # Main road rectangles
    pygame.draw.rect(screen, ASPHALT, (cx - ROAD_W//2, 0, ROAD_W, HEIGHT))
    pygame.draw.rect(screen, ASPHALT, (0, cy - ROAD_W//2, SIM_WIDTH, ROAD_W))

    # Slightly darker intersection box
    box = ROAD_W // 2
    pygame.draw.rect(screen, ASPHALT_DARK,
                     (cx - box, cy - box, ROAD_W, ROAD_W))

    # Crosswalks
    stripe_w, stripe_h, gap = 10, 18, 5
    for i in range(4):
        xo = cx - box + i * (stripe_w + gap)
        pygame.draw.rect(screen, CROSS_STRIPE, (xo, cy - box - stripe_h, stripe_w, stripe_h))
        pygame.draw.rect(screen, CROSS_STRIPE, (xo, cy + box, stripe_w, stripe_h))
    for i in range(4):
        yo = cy - box + i * (stripe_w + gap)
        pygame.draw.rect(screen, CROSS_STRIPE, (cx - box - stripe_h, yo, stripe_h, stripe_w))
        pygame.draw.rect(screen, CROSS_STRIPE, (cx + box, yo, stripe_h, stripe_w))

    # Stop lines
    pygame.draw.rect(screen, LANE_WHITE, (cx,          cy - box - 3, box, 3))
    pygame.draw.rect(screen, LANE_WHITE, (cx - box,    cy + box,     box, 3))
    pygame.draw.rect(screen, LANE_WHITE, (cx - box - 3, cy,          3, box))
    pygame.draw.rect(screen, LANE_WHITE, (cx + box,    cy - box,     3, box))

    # Yellow dashed centre lines (NS road)
    dash_len, dash_gap = 18, 14
    for y in range(0, HEIGHT, dash_len + dash_gap):
        if cy - box > y or y > cy + box:
            pygame.draw.rect(screen, LANE_DASH, (cx - 1, y, 2, dash_len))

    # Yellow dashed centre lines (EW road)
    for x in range(0, SIM_WIDTH, dash_len + dash_gap):
        if cx - box > x or x > cx + box:
            pygame.draw.rect(screen, LANE_DASH, (x, cy - 1, dash_len, 2))


def draw_traffic_light(screen, cx, cy, housing_rect, bulb_positions, states):
    """
    Draw a vertical traffic-light housing with 3 bulbs.
    housing_rect: (x, y, w, h)
    bulb_positions: [(bx, by), ...]  — top=red, mid=yellow, bot=green
    states: (is_red, is_yellow, is_green)
    """
    # pole
    px = housing_rect[0] + housing_rect[2] // 2
    pygame.draw.rect(screen, (30, 32, 40), (px - 2, housing_rect[1] + housing_rect[3], 4, 20))
    # housing
    pygame.draw.rect(screen, TL_HOUSING, housing_rect, border_radius=6)
    pygame.draw.rect(screen, (50, 55, 70), housing_rect, width=1, border_radius=6)
    # bulbs
    bulb_cols = [RED_BULB, YEL_BULB, GRN_BULB]
    for (bx, by), on, col in zip(bulb_positions, states, bulb_cols):
        if on:
            # glow
            gs = pygame.Surface((28, 28), pygame.SRCALPHA)
            pygame.draw.circle(gs, (*col, 50), (14, 14), 13)
            pygame.draw.circle(gs, (*col, 90), (14, 14),  7)
            screen.blit(gs, (bx - 14, by - 14))
            pygame.draw.circle(screen, col, (bx, by), 7)
            pygame.draw.circle(screen, (255, 255, 255), (bx - 2, by - 2), 2)
        else:
            pygame.draw.circle(screen, BULB_OFF, (bx, by), 5)


def draw_lights(screen, light_state):
    """Place 4 traffic-light heads around the intersection."""
    ns_red    = light_state in (1, 3)
    ns_yellow = light_state == 2
    ns_green  = light_state == 0
    ew_red    = light_state in (0, 2)
    ew_yellow = light_state == 3
    ew_green  = light_state == 1

    cx, cy = SIM_WIDTH // 2, HEIGHT // 2

    configs = [
        # (housing_rect,               bulb_positions,   red/yel/grn)
        ((cx + 8,  cy - 110, 20, 62),
         [(cx+18, cy-100), (cx+18, cy-80), (cx+18, cy-60)],
         (ns_red, ns_yellow, ns_green)),

        ((cx - 28, cy + 50, 20, 62),
         [(cx-18, cy+60), (cx-18, cy+80), (cx-18, cy+100)],
         (ns_red, ns_yellow, ns_green)),

        ((cx - 110, cy + 8, 62, 20),
         [(cx-100, cy+18), (cx-80, cy+18), (cx-60, cy+18)],
         (ew_red, ew_yellow, ew_green)),

        ((cx + 50, cy - 28, 62, 20),
         [(cx+60, cy-18), (cx+80, cy-18), (cx+100, cy-18)],
         (ew_red, ew_yellow, ew_green)),
    ]
    for housing, bulbs, states in configs:
        draw_traffic_light(screen, cx, cy, housing, bulbs, states)


def draw_queued_cars(screen, ns_queue, ew_queue):
    cx, cy = SIM_WIDTH // 2, HEIGHT // 2
    for i in range(ns_queue):
        draw_car(screen, cx + 10, cy - 100 - i * 36, True,  CAR_NS)
    for i in range(ew_queue):
        draw_car(screen, cx - 100 - i * 36, cy + 10, False, CAR_EW)


# ─────────────────────────────────────────────────────────────────────────────
# Q-value helpers (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────
def get_q_values(agent, state, mode):
    """Return (q_keep, q_switch) or None."""
    if agent is None or mode not in ("q_learning", "sarsa", "dqn"):
        return None
    ns, ew, light = state
    q_light = min(light, 1)
    if mode in ("q_learning", "sarsa"):
        q = agent.q_table[ns, ew, q_light]
        return float(q[0]), float(q[1])
    else:
        import torch
        sv = agent.preprocess_state(state)
        t  = torch.tensor(sv, dtype=torch.float32).unsqueeze(0).to(agent.device)
        agent.q_net.eval()
        with torch.no_grad():
            out = agent.q_net(t)
        v = out.squeeze().cpu().numpy()
        return float(v[0]), float(v[1])


def get_q_heatmap(agent, light_state, mode, max_queue=5):
    """Return (max_queue+1 x max_queue+1) grid of best Q-value per (ns,ew) or None."""
    if agent is None or mode not in ("q_learning", "sarsa"):
        return None
    q_light = min(light_state, 1)
    size    = max_queue + 1
    grid    = np.zeros((size, size), dtype=np.float32)
    for ns in range(size):
        for ew in range(size):
            grid[ns, ew] = float(np.max(agent.q_table[ns, ew, q_light]))
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard panel  — 3 focused sections
# ─────────────────────────────────────────────────────────────────────────────
def draw_dashboard(screen, fonts, agent, state, mode, max_queue,
                   reward_history, cum_reward, ep_step, max_steps):
    """
    Right-side dashboard.  Three cards, top to bottom:
      1. DECISION — big, coloured, instant readability
      2. REASONING — score comparison (why)
      3. LIVE CHART — rolling reward history (unique feature)
    """
    px   = SIM_WIDTH
    pw   = PANEL_WIDTH
    ph   = HEIGHT
    M    = 14          # left/right margin inside panel
    iw   = pw - M * 2  # inner width

    fxs  = fonts["xs"]
    fsm  = fonts["sm"]
    fmd  = fonts["md"]
    flg  = fonts["lg"]
    fxl  = fonts["xl"]

    ns, ew, light = state
    q_vals = get_q_values(agent, state, mode)

    # ── Background ────────────────────────────────────────────────────────────
    pygame.draw.rect(screen, P_BG, (px, 0, pw, ph))
    pygame.draw.line(screen, P_BORDER, (px, 0), (px, ph), 2)

    # ── Header strip ──────────────────────────────────────────────────────────
    mode_labels = {
        "q_learning": "Q-Learning Agent",
        "sarsa":      "SARSA Agent",
        "dqn":        "DQN Agent",
        "lqf":        "Longest Queue First",
        "fixed":      "Fixed-Time Baseline",
        "random":     "Random Baseline",
    }
    pygame.draw.rect(screen, P_CARD, (px, 0, pw, 44))
    pygame.draw.line(screen, P_BORDER, (px, 44), (px + pw, 44), 1)
    screen.blit(flg.render("Live Dashboard", True, C_CYAN),   (px + M, 8))
    screen.blit(fxs.render(mode_labels.get(mode, mode), True, P_DIM), (px + M, 28))

    y = 52

    # ══════════════════════════════════════════════════════════════════════════
    # CARD 1 — DECISION
    # ══════════════════════════════════════════════════════════════════════════
    CARD_H = 100

    if q_vals is not None:
        q_keep, q_switch = q_vals
        is_switch   = q_switch > q_keep
        adv         = abs(q_switch - q_keep)
        certainty   = min(int(adv / 5.0 * 100), 100)
        dec_text    = "SWITCH LIGHT" if is_switch else "HOLD  LIGHT"
        dec_col     = C_PURPLE if is_switch else C_CYAN
        cert_label  = "HIGH" if certainty > 60 else "MED" if certainty > 25 else "LOW"
    else:
        dec_text, dec_col = "BASELINE", P_DIM
        is_switch, adv, certainty, cert_label = False, 0, 0, ""

    # Card background — subtle tint of the decision colour
    pill(screen, tuple(int(c*0.12) for c in dec_col),
         (px + M, y, iw, CARD_H), r=12, alpha=255)
    pygame.draw.rect(screen, dec_col, (px + M, y, iw, CARD_H), width=2, border_radius=12)

    screen.blit(fxs.render("AGENT DECISION", True, P_DIM), (px + M + 12, y + 10))

    dec_surf = fxl.render(dec_text, True, dec_col)
    screen.blit(dec_surf, (px + M + 12, y + 26))

    if cert_label:
        cert_bg = lc((20, 20, 20), dec_col, 0.3)
        ct = fsm.render(f"Confidence: {cert_label}  ({certainty}%)", True, P_DIM)
        screen.blit(ct, (px + M + 12, y + 74))

    y += CARD_H + 12

    # ══════════════════════════════════════════════════════════════════════════
    # CARD 2 — REASONING
    # ══════════════════════════════════════════════════════════════════════════
    REASON_H = 130

    pill(screen, P_CARD, (px + M, y, iw, REASON_H), r=10)
    pygame.draw.rect(screen, P_BORDER, (px + M, y, iw, REASON_H), width=1, border_radius=10)

    screen.blit(fxs.render("WHY THIS DECISION?", True, P_DIM), (px + M + 12, y + 10))

    if q_vals is not None:
        q_keep, q_switch = q_vals

        for row_i, (lbl, val, col, winner) in enumerate([
            ("Hold  light", q_keep,   C_CYAN,   not is_switch),
            ("Switch light", q_switch, C_PURPLE, is_switch),
        ]):
            ry = y + 28 + row_i * 42

            # Score number — big and clear
            num_s = fmd.render(f"{val:+.2f}", True, col)
            lbl_s = fsm.render(lbl, True, col)
            screen.blit(lbl_s, (px + M + 12, ry))
            screen.blit(num_s, (px + pw - M - num_s.get_width() - 10, ry))

            # Bar — relative width
            bar_y = ry + 18
            bar_w = iw - 24
            pygame.draw.rect(screen, (25, 30, 45), (px + M + 12, bar_y, bar_w, 10), border_radius=4)
            if winner:
                fill = bar_w
            else:
                # Losing bar: show how close it is (50%–95% of full)
                lose_ratio = max(0.5, 1.0 - adv / max(abs(q_keep) + abs(q_switch) + 1e-9, 1.0))
                fill = max(6, int(bar_w * lose_ratio))
            bc = col if winner else lc(col, (60, 30, 50), 0.6)
            pygame.draw.rect(screen, bc, (px + M + 12, bar_y, fill, 10), border_radius=4)
            if winner:
                ws = fxs.render("BEST", True, C_YELLOW)
                screen.blit(ws, (px + pw - M - ws.get_width() - 10, bar_y - 1))

        # Plain-English reason line
        if abs(adv) < 0.05:
            reason = "Nearly equal — agent is uncertain."
        elif is_switch:
            reason = f"Switching scores {adv:.2f} pts higher."
        else:
            reason = f"Holding scores {adv:.2f} pts higher."
        rs = fxs.render(reason, True, P_DIM)
        screen.blit(rs, (px + M + 12, y + REASON_H - 16))

    else:
        nt = fsm.render("Baseline: no Q-scores available.", True, P_DIM)
        screen.blit(nt, (px + M + 12, y + 48))

    y += REASON_H + 12

    # ══════════════════════════════════════════════════════════════════════════
    # CARD 3 — LIVE REWARD CHART  (unique feature)
    # ══════════════════════════════════════════════════════════════════════════
    CHART_LABEL_H = 34
    CHART_BODY_H  = 160
    CARD3_H       = CHART_LABEL_H + CHART_BODY_H + 24

    pill(screen, P_CARD, (px + M, y, iw, CARD3_H), r=10)
    pygame.draw.rect(screen, P_BORDER, (px + M, y, iw, CARD3_H), width=1, border_radius=10)

    screen.blit(fxs.render("REWARD HISTORY  (last 60 steps)", True, P_DIM),
                (px + M + 12, y + 10))
    # Episode stats inline
    ep_s = fxs.render(f"Step {ep_step}/{max_steps}  |  Total: {cum_reward:+.0f}", True, P_DIM)
    screen.blit(ep_s, (px + pw - M - ep_s.get_width() - 10, y + 10))

    # Chart area
    cax = px + M + 12
    cay = y + CHART_LABEL_H
    caw = iw - 24
    cah = CHART_BODY_H

    # Chart background
    pygame.draw.rect(screen, (14, 17, 28), (cax, cay, caw, cah), border_radius=6)

    if len(reward_history) >= 2:
        hist = list(reward_history)
        # Determine y-range with some padding
        lo = min(hist) - 1
        hi = max(hist) + 1
        span = max(hi - lo, 1.0)

        # Horizontal grid lines at 0 and halfway marks
        zero_y = cay + cah - int((0 - lo) / span * cah)
        zero_y = max(cay, min(cay + cah, zero_y))

        for frac in [0.25, 0.5, 0.75]:
            gy = cay + int(frac * cah)
            pygame.draw.line(screen, (30, 36, 55), (cax, gy), (cax + caw, gy))

        # Zero line (slightly brighter)
        pygame.draw.line(screen, (55, 62, 85), (cax, zero_y), (cax + caw, zero_y))

        # Plot the line
        n   = len(hist)
        pts = []
        for i, val in enumerate(hist):
            px_ = cax + int(i / max(n - 1, 1) * caw)
            py_ = cay + cah - int((val - lo) / span * cah)
            py_ = max(cay + 1, min(cay + cah - 1, py_))
            pts.append((px_, py_))

        # Filled area under the line
        if len(pts) >= 2:
            poly = [pts[0]] + pts + [(pts[-1][0], cay + cah), (pts[0][0], cay + cah)]
            fill_surf = pygame.Surface((caw + 30, cah + 10), pygame.SRCALPHA)
            shifted = [(p[0] - cax, p[1] - cay) for p in poly]
            pygame.draw.polygon(fill_surf, (*C_CYAN, 25), shifted)
            screen.blit(fill_surf, (cax, cay))

            # Line itself
            if len(pts) >= 2:
                pygame.draw.lines(screen, C_CYAN, False, pts, 2)

            # Current point dot
            pygame.draw.circle(screen, C_CYAN, pts[-1], 4)
            pygame.draw.circle(screen, P_BG,   pts[-1], 2)

        # Y-axis labels
        lo_s = fxs.render(f"{lo:.0f}", True, P_DIM)
        hi_s = fxs.render(f"{hi:.0f}", True, P_DIM)
        screen.blit(hi_s, (cax + 2, cay + 2))
        screen.blit(lo_s, (cax + 2, cay + cah - 13))

    else:
        nt = fsm.render("Collecting data…", True, P_DIM)
        screen.blit(nt, (cax + caw//2 - nt.get_width()//2, cay + cah//2 - 8))

    y += CARD3_H + 12

    # ── Compact state info row (below the cards) ───────────────────────────────
    pygame.draw.line(screen, P_BORDER, (px + M, y), (px + pw - M, y), 1)
    y += 8

    light_names = {0: "NS Green", 1: "EW Green", 2: "NS Yellow", 3: "EW Yellow"}
    light_cols  = {0: GRN_BULB, 1: GRN_BULB, 2: YEL_BULB, 3: YEL_BULB}
    lc_text = light_cols.get(light, P_DIM)

    info_parts = [
        (f"NS: {ns}/{max_queue}", lc(C_CYAN, (180,180,180), ns/max(max_queue,1))),
        (f"EW: {ew}/{max_queue}", lc(C_RED,  (180,180,180), ew/max(max_queue,1))),
        (light_names.get(light, "?"), lc_text),
    ]
    xi = px + M
    for txt, col in info_parts:
        s = fsm.render(txt, True, col)
        screen.blit(s, (xi, y))
        xi += s.get_width() + 18

    # ── Footer ────────────────────────────────────────────────────────────────
    footer_y = ph - 20
    pygame.draw.line(screen, P_BORDER, (px + M, footer_y - 4), (px + pw - M, footer_y - 4), 1)
    screen.blit(fxs.render("Cyan = Hold   Purple = Switch", True, (50, 56, 75)),
                (px + M, footer_y + 1))


# ─────────────────────────────────────────────────────────────────────────────
# Top HUD (smaller, stays out of the way)
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(screen, font, ns, ew, agent_action, mode_name, phase_name, rates):
    surf = pygame.Surface((320, 68), pygame.SRCALPHA)
    surf.fill((12, 14, 22, 200))
    screen.blit(surf, (16, 16))
    pygame.draw.rect(screen, P_BORDER, (16, 16, 320, 68), width=1, border_radius=6)

    bold = pygame.font.SysFont("Consolas", 14, bold=True)
    norm = pygame.font.SysFont("Consolas", 13)

    screen.blit(bold.render(mode_name, True, C_YELLOW), (28, 24))
    screen.blit(norm.render(f"Phase: {phase_name}  (NS {int(rates[0]*100)}%  EW {int(rates[1]*100)}%)",
                            True, (170, 175, 190)), (28, 42))
    screen.blit(norm.render(f"Queue  NS:{ns}  EW:{ew}   Action: {agent_action}",
                            True, (200, 205, 215)), (28, 58))


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def run_visualization(agent, env, mode):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL Traffic Signal Control — Live Visualizer")
    clock = pygame.time.Clock()

    fonts = {
        "xs":  pygame.font.SysFont("Consolas", 12),
        "sm":  pygame.font.SysFont("Consolas", 14, bold=True),
        "md":  pygame.font.SysFont("Consolas", 16),
        "lg":  pygame.font.SysFont("Consolas", 19, bold=True),
        "xl":  pygame.font.SysFont("Consolas", 23, bold=True),
    }
    font_hud = pygame.font.SysFont("Consolas", 13)

    state       = env.reset()
    ns, ew, light = state
    agent_action  = "STARTING..."
    cum_reward    = 0.0
    ep_step       = 0
    max_steps     = env.max_steps

    # Rolling buffer: store per-step rewards for the chart
    CHART_LEN    = 60
    reward_hist  = deque(maxlen=CHART_LEN)

    animating_cars   = []
    last_logic_time  = pygame.time.get_ticks()

    mode_names = {
        "q_learning": "Trained Q-Learning Agent",
        "sarsa":      "Trained SARSA Agent",
        "dqn":        "Trained DQN Agent",
        "lqf":        "Longest Queue First",
        "fixed":      "Fixed-Time (5 steps)",
        "random":     "Random Switch",
    }
    mode_name = mode_names.get(mode, mode.upper())

    while True:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if now - last_logic_time > LOGIC_DELAY_MS:
            prev_ns, prev_ew, prev_light = state

            if prev_light in (2, 3):
                raw_action   = 0
                agent_action = "YELLOW PHASE"
            else:
                if mode in ("q_learning", "sarsa", "dqn") and agent is not None:
                    raw_action = agent.choose_action(state, epsilon=0.0)
                elif mode == "lqf":
                    raw_action = (1 if (prev_light == 0 and prev_ew > prev_ns
                                       and env.time_since_switch >= 3)
                                     or (prev_light == 1 and prev_ns > prev_ew
                                         and env.time_since_switch >= 3)
                                  else 0)
                elif mode == "fixed":
                    raw_action = 1 if env.steps > 0 and env.steps % 5 == 0 else 0
                else:
                    raw_action = random.choice([0, 1])

                agent_action = "SWITCH" if raw_action == 1 else "HOLD"

            state, reward, done, _ = env.step(raw_action)
            cum_reward  += reward
            ep_step     += 1
            reward_hist.append(reward)

            if done:
                state      = env.reset()
                cum_reward = 0.0
                ep_step    = 0

            # Spawn an animated crossing car
            cx, cy = SIM_WIDTH // 2, HEIGHT // 2
            if prev_light in (0, 3) and prev_ns > 0:
                animating_cars.append(AnimatedCar(cx + 10, cy - 110, 0, 9, CAR_NS, True))
            elif prev_light in (1, 2) and prev_ew > 0:
                animating_cars.append(AnimatedCar(cx - 110, cy + 10, 9, 0, CAR_EW, False))

            ns, ew, light = state
            last_logic_time = now

        for car in animating_cars:
            car.move()
        animating_cars = [c for c in animating_cars
                          if -60 < c.x < SIM_WIDTH + 60 and -60 < c.y < HEIGHT + 60]

        # ── Render ────────────────────────────────────────────────────────────
        draw_road(screen)
        draw_queued_cars(screen, ns, ew)
        for car in animating_cars:
            car.draw(screen)
        draw_lights(screen, light)

        phase_name = env.get_traffic_phase()
        rates      = env.get_arrival_probabilities()
        draw_hud(screen, font_hud, ns, ew, agent_action, mode_name, phase_name, rates)

        draw_dashboard(screen, fonts, agent, state, mode, env.max_queue,
                       reward_hist, cum_reward, ep_step, max_steps)

        pygame.display.flip()
        clock.tick(RENDER_FPS)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="RL Traffic Signal Visualizer")
    parser.add_argument("--mode", default="q_learning",
                        choices=["q_learning", "sarsa", "dqn", "lqf", "fixed", "random"])
    args = parser.parse_args()

    env   = TrafficEnv(max_queue=5, max_steps=100)
    agent = None

    if args.mode == "q_learning":
        agent = QLearningAgent(max_queue=5)
        if os.path.exists("q_table_qlearning.npy"):
            print("Loading trained Q-table for Q-Learning...")
            agent.q_table = np.load("q_table_qlearning.npy")
        else:
            print("WARNING: No trained Q-table found. Running untrained agent.")

    elif args.mode == "sarsa":
        agent = SARSAAgent(max_queue=5)
        if os.path.exists("q_table_sarsa.npy"):
            print("Loading trained Q-table for SARSA...")
            agent.q_table = np.load("q_table_sarsa.npy")
        else:
            print("WARNING: No trained SARSA Q-table found.")

    elif args.mode == "dqn":
        try:
            from dqn_agent import DQNAgent
            agent = DQNAgent(state_size=6, action_size=2, max_queue=5)
            if os.path.exists("dqn_model.pth"):
                print("Loading trained DQN weights...")
                import torch
                agent.q_net.load_state_dict(torch.load("dqn_model.pth", map_location=agent.device))
                agent.q_net.eval()
                print("DQN weights loaded successfully.")
            else:
                print("WARNING: No trained DQN model found.")
        except ImportError:
            print("DQN agent not available.")

    print(f"\nLaunching visualizer — mode: {args.mode.upper()}")
    run_visualization(agent, env, args.mode)