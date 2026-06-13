import pygame
import sys
import numpy as np
import random
from environment import TrafficEnv
from agent import QLearningAgent, SARSAAgent

# Layout Constants
WIDTH, HEIGHT = 800, 600
RENDER_FPS = 60
LOGIC_DELAY_MS = 600

# Sleek Cyberpunk/Dark-Mode Color Palette
GRASS_GREEN = (22, 34, 28)     # Soft dark forest slate
ROAD_GRAY = (38, 40, 46)        # Modern asphalt grey
LINE_YELLOW = (235, 165, 35)    # Rich amber lane divider
LINE_WHITE = (225, 225, 230)    # Soft white stop line
PED_STRIPE = (100, 104, 115)    # Slate grey for crosswalk stripes
RED_LIGHT = (239, 83, 80)       # Glowing red
GREEN_LIGHT = (102, 187, 106)   # Glowing green
YELLOW_LIGHT = (255, 202, 40)    # Glowing gold yellow
BULB_OFF = (45, 45, 45)         # Dark grey bulb off
HOUSING_BLACK = (25, 25, 25)    # Dark traffic light casing
CAR_NS_COLOR = (30, 136, 229)   # Vibrant ocean blue
CAR_EW_COLOR = (229, 57, 53)    # Vibrant crimson red
GLASS_COLOR = (128, 222, 234)   # Glowing cyan glass
TEXT_WHITE = (245, 245, 245)

def draw_sleek_car(screen, x, y, is_ns, base_color):
    h_var = int((x + y) % 15) - 7
    color = (
        max(0, min(255, base_color[0] + h_var * 3)),
        max(0, min(255, base_color[1] + h_var * 3)),
        max(0, min(255, base_color[2] + h_var * 3))
    )
    
    if is_ns:  # Moves down (Southbound)
        w, h = 26, 30
        pygame.draw.rect(screen, color, (x, y, w, h), border_radius=5)
        pygame.draw.rect(screen, GLASS_COLOR, (x + 3, y + 8, w - 6, 6), border_radius=1)
        pygame.draw.rect(screen, GLASS_COLOR, (x + 3, y + h - 6, w - 6, 3), border_radius=1)
        pygame.draw.circle(screen, (255, 245, 180), (x + 5, y + h - 2), 3)
        pygame.draw.circle(screen, (255, 245, 180), (x + w - 5, y + h - 2), 3)
        pygame.draw.rect(screen, (220, 20, 20), (x + 3, y + 1, 4, 2))
        pygame.draw.rect(screen, (220, 20, 20), (x + w - 7, y + 1, 4, 2))
    else:  # Moves right (Eastbound)
        w, h = 30, 26
        pygame.draw.rect(screen, color, (x, y, w, h), border_radius=5)
        pygame.draw.rect(screen, GLASS_COLOR, (x + w - 14, y + 3, 6, h - 6), border_radius=1)
        pygame.draw.rect(screen, GLASS_COLOR, (x + 3, y + 3, 3, h - 6), border_radius=1)
        pygame.draw.circle(screen, (255, 245, 180), (x + w - 2, y + 5), 3)
        pygame.draw.circle(screen, (255, 245, 180), (x + w - 2, y + h - 5), 3)
        pygame.draw.rect(screen, (220, 20, 20), (x + 1, y + 3, 2, 4))
        pygame.draw.rect(screen, (220, 20, 20), (x + 1, y + h - 7, 2, 4))

class AnimatedCar:
    def __init__(self, x, y, dx, dy, color, is_ns):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.color = color
        self.is_ns = is_ns

    def move(self):
        self.x += self.dx
        self.y += self.dy

    def draw(self, screen):
        draw_sleek_car(screen, self.x, self.y, self.is_ns, self.color)

def draw_base_roads(screen):
    screen.fill(GRASS_GREEN)
    pygame.draw.rect(screen, ROAD_GRAY, (WIDTH//2 - 50, 0, 100, HEIGHT))  
    pygame.draw.rect(screen, ROAD_GRAY, (0, HEIGHT//2 - 50, WIDTH, 100))  
    
    # Zebra crosswalk stripes
    for x in range(WIDTH//2 - 45, WIDTH//2 + 45, 15):
        pygame.draw.rect(screen, PED_STRIPE, (x, HEIGHT//2 - 70, 8, 16), border_radius=1)
        pygame.draw.rect(screen, PED_STRIPE, (x, HEIGHT//2 + 54, 8, 16), border_radius=1)
    for y in range(HEIGHT//2 - 45, HEIGHT//2 + 45, 15):
        pygame.draw.rect(screen, PED_STRIPE, (WIDTH//2 - 70, y, 16, 8), border_radius=1)
        pygame.draw.rect(screen, PED_STRIPE, (WIDTH//2 + 54, y, 16, 8), border_radius=1)

    # Stop lines
    pygame.draw.rect(screen, LINE_WHITE, (WIDTH//2, HEIGHT//2 - 53, 50, 3))
    pygame.draw.rect(screen, LINE_WHITE, (WIDTH//2 - 50, HEIGHT//2 + 50, 50, 3))
    pygame.draw.rect(screen, LINE_WHITE, (WIDTH//2 - 53, HEIGHT//2, 3, 50))
    pygame.draw.rect(screen, LINE_WHITE, (WIDTH//2 + 50, HEIGHT//2 - 50, 3, 50))

    # Center Lane Dividers
    for y in range(0, HEIGHT, 30):
        if y < HEIGHT//2 - 50 or y > HEIGHT//2 + 50:
            pygame.draw.rect(screen, LINE_YELLOW, (WIDTH//2 - 1, y + 5, 2, 15))
    for x in range(0, WIDTH, 30):
        if x < WIDTH//2 - 50 or x > WIDTH//2 + 50:
            pygame.draw.rect(screen, LINE_YELLOW, (x + 5, HEIGHT//2 - 1, 15, 2))

def draw_light_bulb(screen, center, color, is_on):
    if is_on:
        glow_surf = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (color[0], color[1], color[2], 45), (16, 16), 14)
        pygame.draw.circle(glow_surf, (color[0], color[1], color[2], 100), (16, 16), 8)
        screen.blit(glow_surf, (center[0] - 16, center[1] - 16))
        pygame.draw.circle(screen, color, center, 4)
        pygame.draw.circle(screen, (255, 255, 255), (center[0] - 1.5, center[1] - 1.5), 1.5)
    else:
        pygame.draw.circle(screen, BULB_OFF, center, 4)

def draw_overhead_lights(screen, light_state):
    ns_red = (light_state == 1 or light_state == 3)
    ns_yellow = (light_state == 2)
    ns_green = (light_state == 0)
    
    ew_red = (light_state == 0 or light_state == 2)
    ew_yellow = (light_state == 3)
    ew_green = (light_state == 1)
    
    pygame.draw.rect(screen, HOUSING_BLACK, (WIDTH//2 + 5, HEIGHT//2 - 70, 54, 16), border_radius=4)
    draw_light_bulb(screen, (WIDTH//2 + 14, HEIGHT//2 - 62), RED_LIGHT, ns_red)
    draw_light_bulb(screen, (WIDTH//2 + 32, HEIGHT//2 - 62), YELLOW_LIGHT, ns_yellow)
    draw_light_bulb(screen, (WIDTH//2 + 50, HEIGHT//2 - 62), GREEN_LIGHT, ns_green)
    
    pygame.draw.rect(screen, HOUSING_BLACK, (WIDTH//2 - 59, HEIGHT//2 + 55, 54, 16), border_radius=4)
    draw_light_bulb(screen, (WIDTH//2 - 50, HEIGHT//2 + 63), RED_LIGHT, ns_red)
    draw_light_bulb(screen, (WIDTH//2 - 32, HEIGHT//2 + 63), YELLOW_LIGHT, ns_yellow)
    draw_light_bulb(screen, (WIDTH//2 - 14, HEIGHT//2 + 63), GREEN_LIGHT, ns_green)
    
    pygame.draw.rect(screen, HOUSING_BLACK, (WIDTH//2 - 70, HEIGHT//2 + 5, 16, 54), border_radius=4)
    draw_light_bulb(screen, (WIDTH//2 - 62, HEIGHT//2 + 14), RED_LIGHT, ew_red)
    draw_light_bulb(screen, (WIDTH//2 - 62, HEIGHT//2 + 32), YELLOW_LIGHT, ew_yellow)
    draw_light_bulb(screen, (WIDTH//2 - 62, HEIGHT//2 + 50), GREEN_LIGHT, ew_green)
    
    pygame.draw.rect(screen, HOUSING_BLACK, (WIDTH//2 + 55, HEIGHT//2 - 59, 16, 54), border_radius=4)
    draw_light_bulb(screen, (WIDTH//2 + 63, HEIGHT//2 - 50), RED_LIGHT, ew_red)
    draw_light_bulb(screen, (WIDTH//2 + 63, HEIGHT//2 - 32), YELLOW_LIGHT, ew_yellow)
    draw_light_bulb(screen, (WIDTH//2 + 63, HEIGHT//2 - 14), GREEN_LIGHT, ew_green)

def draw_queued_cars(screen, ns_queue, ew_queue):
    for i in range(ns_queue):
        draw_sleek_car(screen, WIDTH//2 + 12, HEIGHT//2 - 85 - (i * 38), True, CAR_NS_COLOR)
    for i in range(ew_queue):
        draw_sleek_car(screen, WIDTH//2 - 85 - (i * 38), HEIGHT//2 + 12, False, CAR_EW_COLOR)

def draw_hud(screen, font, ns, ew, agent_action, steps, human_score, ai_score, phase_name, rates):
    hud_surface = pygame.Surface((380, 185), pygame.SRCALPHA)
    hud_surface.fill((20, 20, 20, 220)) 
    screen.blit(hud_surface, (20, 20))
    pygame.draw.rect(screen, LINE_WHITE, (20, 20, 380, 185), width=1, border_radius=4)
    
    title_font = pygame.font.SysFont(None, 22, bold=True)
    highlight_font = pygame.font.SysFont(None, 20, bold=True)
    
    screen.blit(title_font.render("MODE: Interactive Manual Control", True, LINE_YELLOW), (35, 30))
    screen.blit(font.render(f"PHASE: {phase_name} (NS: {int(rates[0]*100)}%, EW: {int(rates[1]*100)}%)", True, LINE_WHITE), (35, 55))
    screen.blit(font.render(f"NS Lane Queue: {ns}/5  |  EW Lane Queue: {ew}/5", True, TEXT_WHITE), (35, 80))
    screen.blit(font.render(f"Step: {steps}/100 | Press SPACE to switch light!", True, TEXT_WHITE), (35, 105))
    
    # Score Comparison
    screen.blit(highlight_font.render(f"YOUR SCORE (Penalty): {int(human_score)}", True, (244, 67, 54) if human_score < ai_score else (76, 175, 80)), (35, 135))
    screen.blit(highlight_font.render(f"AI AGENT SCORE (Live): {int(ai_score)}", True, LINE_YELLOW), (35, 155))

def draw_game_over(screen, human_score, ai_score):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((10, 10, 10, 230))
    screen.blit(overlay, (0, 0))
    
    title_font = pygame.font.SysFont(None, 48, bold=True)
    body_font = pygame.font.SysFont(None, 28)
    
    result_text = "Congratulations! You Beat the AI!" if human_score > ai_score else "AI Wins! Try again to optimize traffic!"
    result_color = GREEN_LIGHT if human_score > ai_score else RED_LIGHT
    if human_score == ai_score:
        result_text = "It's a Tie!"
        result_color = YELLOW_LIGHT
        
    screen.blit(title_font.render("EPISODE COMPLETED", True, TEXT_WHITE), (WIDTH//2 - 200, HEIGHT//2 - 120))
    screen.blit(body_font.render(result_text, True, result_color), (WIDTH//2 - 180, HEIGHT//2 - 50))
    
    screen.blit(body_font.render(f"Your Score: {int(human_score)}", True, TEXT_WHITE), (WIDTH//2 - 100, HEIGHT//2 + 10))
    screen.blit(body_font.render(f"AI Score: {int(ai_score)}", True, TEXT_WHITE), (WIDTH//2 - 100, HEIGHT//2 + 40))
    
    screen.blit(body_font.render("Press 'R' to Restart the Episode", True, LINE_YELLOW), (WIDTH//2 - 150, HEIGHT//2 + 100))

def run_interactive_visualization(ai_agent):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Interactive RL Traffic Signal Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)
    
    # We maintain two identical environments, seeded to have the exact same starting queues
    start_seed = random.randint(0, 10000)
    
    def reset_envs(seed):
        random.seed(seed)
        np.random.seed(seed)
        env_h = TrafficEnv(max_queue=5, max_steps=100)
        env_a = TrafficEnv(max_queue=5, max_steps=100)
        
        # Enforce exact same initial state
        init_state = env_h.reset()
        env_a.reset()
        env_a.ns_queue = env_h.ns_queue
        env_a.ew_queue = env_h.ew_queue
        env_a.green_light = env_h.green_light
        env_a.time_since_switch = env_h.time_since_switch
        
        return env_h, env_a, init_state
        
    env_h, env_a, state_h = reset_envs(start_seed)
    state_a = state_h
    
    human_score = 0.0
    ai_score = 0.0
    
    agent_action_desc = "INITIALIZING..."
    user_action = 0  # 0: Keep, 1: Switch
    
    animating_cars = []
    last_logic_time = pygame.time.get_ticks()
    
    game_over = False
    
    while True:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not game_over:
                    # Queue a switch action for the human
                    # (only gets consumed if current light is active green)
                    if env_h.green_light not in [2, 3] and env_h.time_since_switch >= 3:
                        user_action = 1
                elif event.key == pygame.K_r:
                    # Restart
                    start_seed = random.randint(0, 10000)
                    env_h, env_a, state_h = reset_envs(start_seed)
                    state_a = state_h
                    human_score = 0.0
                    ai_score = 0.0
                    animating_cars = []
                    last_logic_time = current_time
                    game_over = False
                    user_action = 0
                    agent_action_desc = "INITIALIZING..."
                    
        if not game_over and (current_time - last_logic_time > LOGIC_DELAY_MS):
            prev_ns_h, prev_ew_h, prev_light_h = state_h
            prev_ns_a, prev_ew_a, prev_light_a = state_a
            
            # --- Human Environment Decision Step ---
            # Automatically transition if in yellow phase
            if prev_light_h in [2, 3]:
                raw_user_action = 0
            else:
                raw_user_action = user_action
                user_action = 0  # Reset
                
            # Advance human environment
            # Save random state so both environments spawn same arrivals if actions are same
            # Since random transitions can occur, we match the arrival outcomes by seeding before step
            rand_state = random.getstate()
            np_rand_state = np.random.get_state()
            
            state_h, reward_h, done_h, info_h = env_h.step(raw_user_action)
            human_score += reward_h
            
            # --- AI Environment Decision Step ---
            # Restore random state so arrivals are simulated identically
            random.setstate(rand_state)
            np.random.set_state(np_rand_state)
            
            if prev_light_a in [2, 3]:
                raw_ai_action = 0
                agent_action_desc = "YELLOW TRANSITION"
            else:
                raw_ai_action = ai_agent.choose_action(state_a, epsilon=0.0)
                agent_action_desc = "SWITCH LIGHT" if raw_ai_action == 1 else "HOLD LIGHT"
                
            state_a, reward_a, done_a, info_a = env_a.step(raw_ai_action)
            ai_score += reward_a
            
            # Add animating cars for human view
            if (prev_light_h == 0 or prev_light_h == 3) and prev_ns_h > 0:
                animating_cars.append(AnimatedCar(WIDTH//2 + 12, HEIGHT//2 - 85, 0, 8, CAR_NS_COLOR, True))
            elif (prev_light_h == 1 or prev_light_h == 2) and prev_ew_h > 0:
                animating_cars.append(AnimatedCar(WIDTH//2 - 85, HEIGHT//2 + 12, 8, 0, CAR_EW_COLOR, False))
                
            last_logic_time = current_time
            if done_h:
                game_over = True
                
        # Update animations
        if not game_over:
            for car in animating_cars:
                car.move()
            animating_cars = [car for car in animating_cars if -50 < car.x < WIDTH + 50 and -50 < car.y < HEIGHT + 50]
            
        # Draw everything (Human View)
        draw_base_roads(screen)
        draw_queued_cars(screen, state_h[0], state_h[1])
        for car in animating_cars:
            car.draw(screen)
        draw_overhead_lights(screen, state_h[2])
        
        # Get phase and rates for HUD
        phase_name = env_h.get_traffic_phase()
        rates = env_h.get_arrival_probabilities()
        draw_hud(screen, font, state_h[0], state_h[1], agent_action_desc, env_h.steps, human_score, ai_score, phase_name, rates)
        
        if game_over:
            draw_game_over(screen, human_score, ai_score)
            
        pygame.display.flip()
        clock.tick(RENDER_FPS)

if __name__ == "__main__":
    import os
    
    # Load trained SARSA agent as AI opponent (has the highest average reward)
    ai_agent = SARSAAgent(max_queue=5)
    if os.path.exists("q_table_sarsa.npy"):
        print("Loading trained Q-table for SARSA agent...")
        ai_agent.q_table = np.load("q_table_sarsa.npy")
    else:
        print("Trained Q-table for SARSA not found! Training a fresh agent first (500 episodes)...")
        from compare_algorithms import train_sarsa
        env = TrafficEnv(max_queue=5, max_steps=100)
        train_sarsa(env, ai_agent, num_episodes=500)
        np.save("q_table_sarsa.npy", ai_agent.q_table)
        
    print("\nLaunching Interactive Traffic Visualizer Game...")
    print("Press SPACEBAR to switch the signals. Press 'R' to reset the episode.")
    run_interactive_visualization(ai_agent)
