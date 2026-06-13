import pygame
import sys
import numpy as np
from environment import TrafficEnv
from agent import QLearningAgent, SARSAAgent
from main import train_agent

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
    # Add subtle color variation based on position so traffic looks diverse
    h_var = int((x + y) % 15) - 7
    color = (
        max(0, min(255, base_color[0] + h_var * 3)),
        max(0, min(255, base_color[1] + h_var * 3)),
        max(0, min(255, base_color[2] + h_var * 3))
    )
    
    if is_ns:  # Moves down (Southbound)
        w, h = 26, 30
        # Main body
        pygame.draw.rect(screen, color, (x, y, w, h), border_radius=5)
        # Windshield
        pygame.draw.rect(screen, GLASS_COLOR, (x + 3, y + 8, w - 6, 6), border_radius=1)
        # Rear window
        pygame.draw.rect(screen, GLASS_COLOR, (x + 3, y + h - 6, w - 6, 3), border_radius=1)
        # Headlights (front, bottom side)
        pygame.draw.circle(screen, (255, 245, 180), (x + 5, y + h - 2), 3)
        pygame.draw.circle(screen, (255, 245, 180), (x + w - 5, y + h - 2), 3)
        # Taillights (rear, top side)
        pygame.draw.rect(screen, (220, 20, 20), (x + 3, y + 1, 4, 2))
        pygame.draw.rect(screen, (220, 20, 20), (x + w - 7, y + 1, 4, 2))
    else:  # Moves right (Eastbound)
        w, h = 30, 26
        # Main body
        pygame.draw.rect(screen, color, (x, y, w, h), border_radius=5)
        # Windshield
        pygame.draw.rect(screen, GLASS_COLOR, (x + w - 14, y + 3, 6, h - 6), border_radius=1)
        # Rear window
        pygame.draw.rect(screen, GLASS_COLOR, (x + 3, y + 3, 3, h - 6), border_radius=1)
        # Headlights (front, right side)
        pygame.draw.circle(screen, (255, 245, 180), (x + w - 2, y + 5), 3)
        pygame.draw.circle(screen, (255, 245, 180), (x + w - 2, y + h - 5), 3)
        # Taillights (rear, left side)
        pygame.draw.rect(screen, (220, 20, 20), (x + 1, y + 3, 2, 4))
        pygame.draw.rect(screen, (220, 20, 20), (x + 1, y + h - 7, 2, 4))


class AnimatedCar:
    """Handles the smooth movement of cars crossing the intersection."""
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
    """Draws ground layer: grass, roads, lane divides, stop lines, and zebra crosswalks."""
    screen.fill(GRASS_GREEN)
    
    # Asphalt
    pygame.draw.rect(screen, ROAD_GRAY, (WIDTH//2 - 50, 0, 100, HEIGHT))  
    pygame.draw.rect(screen, ROAD_GRAY, (0, HEIGHT//2 - 50, WIDTH, 100))  
    
    # Pedestrian Crosswalks (Zebra stripes)
    # North
    for x in range(WIDTH//2 - 45, WIDTH//2 + 45, 15):
        pygame.draw.rect(screen, PED_STRIPE, (x, HEIGHT//2 - 70, 8, 16), border_radius=1)
    # South
    for x in range(WIDTH//2 - 45, WIDTH//2 + 45, 15):
        pygame.draw.rect(screen, PED_STRIPE, (x, HEIGHT//2 + 54, 8, 16), border_radius=1)
    # West
    for y in range(HEIGHT//2 - 45, HEIGHT//2 + 45, 15):
        pygame.draw.rect(screen, PED_STRIPE, (WIDTH//2 - 70, y, 16, 8), border_radius=1)
    # East
    for y in range(HEIGHT//2 - 45, HEIGHT//2 + 45, 15):
        pygame.draw.rect(screen, PED_STRIPE, (WIDTH//2 + 54, y, 16, 8), border_radius=1)

    # Stop Lines
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
    """Draws a traffic light bulb, applying a glowing bloom effect if it is on."""
    if is_on:
        # Glow layers using alpha transparency surface
        glow_surf = pygame.Surface((32, 32), pygame.SRCALPHA)
        # Outer soft glow
        pygame.draw.circle(glow_surf, (color[0], color[1], color[2], 45), (16, 16), 14)
        # Inner glow
        pygame.draw.circle(glow_surf, (color[0], color[1], color[2], 100), (16, 16), 8)
        screen.blit(glow_surf, (center[0] - 16, center[1] - 16))
        # Solid core
        pygame.draw.circle(screen, color, center, 4)
        # Highlight shininess
        pygame.draw.circle(screen, (255, 255, 255), (center[0] - 1.5, center[1] - 1.5), 1.5)
    else:
        pygame.draw.circle(screen, BULB_OFF, center, 4)


def draw_overhead_lights(screen, light_state):
    """Draws the overhead gantries with 3-bulb traffic lights."""
    # Light states:
    # 0: NS Green, EW Red
    # 1: EW Green, NS Red
    # 2: NS Yellow, EW Red
    # 3: EW Yellow, NS Red
    
    # NS bulb statuses
    ns_red = (light_state == 1 or light_state == 3)
    ns_yellow = (light_state == 2)
    ns_green = (light_state == 0)
    
    # EW bulb statuses
    ew_red = (light_state == 0 or light_state == 2)
    ew_yellow = (light_state == 3)
    ew_green = (light_state == 1)
    
    # --- North Incoming Lane (facing down) ---
    # Casing size: 54 wide, 16 high
    pygame.draw.rect(screen, HOUSING_BLACK, (WIDTH//2 + 5, HEIGHT//2 - 70, 54, 16), border_radius=4)
    # Bulbs: Red, Yellow, Green
    draw_light_bulb(screen, (WIDTH//2 + 14, HEIGHT//2 - 62), RED_LIGHT, ns_red)
    draw_light_bulb(screen, (WIDTH//2 + 32, HEIGHT//2 - 62), YELLOW_LIGHT, ns_yellow)
    draw_light_bulb(screen, (WIDTH//2 + 50, HEIGHT//2 - 62), GREEN_LIGHT, ns_green)
    
    # --- South Incoming Lane (facing up) ---
    pygame.draw.rect(screen, HOUSING_BLACK, (WIDTH//2 - 59, HEIGHT//2 + 55, 54, 16), border_radius=4)
    draw_light_bulb(screen, (WIDTH//2 - 50, HEIGHT//2 + 63), RED_LIGHT, ns_red)
    draw_light_bulb(screen, (WIDTH//2 - 32, HEIGHT//2 + 63), YELLOW_LIGHT, ns_yellow)
    draw_light_bulb(screen, (WIDTH//2 - 14, HEIGHT//2 + 63), GREEN_LIGHT, ns_green)
    
    # --- West Incoming Lane (facing right) ---
    # Casing size: 16 wide, 54 high
    pygame.draw.rect(screen, HOUSING_BLACK, (WIDTH//2 - 70, HEIGHT//2 + 5, 16, 54), border_radius=4)
    # Bulbs: Red, Yellow, Green (vertical top to bottom)
    draw_light_bulb(screen, (WIDTH//2 - 62, HEIGHT//2 + 14), RED_LIGHT, ew_red)
    draw_light_bulb(screen, (WIDTH//2 - 62, HEIGHT//2 + 32), YELLOW_LIGHT, ew_yellow)
    draw_light_bulb(screen, (WIDTH//2 - 62, HEIGHT//2 + 50), GREEN_LIGHT, ew_green)
    
    # --- East Incoming Lane (facing left) ---
    pygame.draw.rect(screen, HOUSING_BLACK, (WIDTH//2 + 55, HEIGHT//2 - 59, 16, 54), border_radius=4)
    draw_light_bulb(screen, (WIDTH//2 + 63, HEIGHT//2 - 50), RED_LIGHT, ew_red)
    draw_light_bulb(screen, (WIDTH//2 + 63, HEIGHT//2 - 32), YELLOW_LIGHT, ew_yellow)
    draw_light_bulb(screen, (WIDTH//2 + 63, HEIGHT//2 - 14), GREEN_LIGHT, ew_green)


def draw_queued_cars(screen, ns_queue, ew_queue):
    # Centered N/S cars
    for i in range(ns_queue):
        x = WIDTH//2 + 12
        y = HEIGHT//2 - 85 - (i * 38)
        draw_sleek_car(screen, x, y, True, CAR_NS_COLOR)
        
    # Centered E/W cars
    for i in range(ew_queue):
        x = WIDTH//2 - 85 - (i * 38)
        y = HEIGHT//2 + 12
        draw_sleek_car(screen, x, y, False, CAR_EW_COLOR)


def draw_hud(screen, font, ns, ew, agent_action, mode_name):
    hud_surface = pygame.Surface((340, 110), pygame.SRCALPHA)
    hud_surface.fill((20, 20, 20, 220)) 
    screen.blit(hud_surface, (20, 20))
    pygame.draw.rect(screen, LINE_WHITE, (20, 20, 340, 110), width=1, border_radius=4)
    
    title_font = pygame.font.SysFont(None, 22, bold=True)
    screen.blit(title_font.render(f"MODE: {mode_name}", True, LINE_YELLOW), (35, 30))
    screen.blit(font.render(f"NS Lane Queue: {ns}/5  |  EW Lane Queue: {ew}/5", True, TEXT_WHITE), (35, 60))
    screen.blit(font.render(f"Decision: {agent_action}", True, TEXT_WHITE), (35, 90))


def run_visualization(agent, env, mode):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Reinforcement Learning Traffic Signal Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)
    
    state = env.reset()
    ns, ew, light = state
    agent_action = "INITIALIZING..."
    
    animating_cars = []
    last_logic_time = pygame.time.get_ticks()
    
    # Human readable mode name
    mode_names = {
        "q_learning": "Trained Q-Learning Agent",
        "sarsa": "Trained SARSA Agent",
        "dqn": "Trained DQN Agent",
        "lqf": "Longest Queue First Heuristic",
        "fixed": "Fixed-Time Switch (5-steps)",
        "random": "Random Switch"
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
                # Action selection based on selected mode
                if mode in ["q_learning", "sarsa", "dqn"] and agent is not None:
                    raw_action = agent.choose_action(state, epsilon=0.0)
                elif mode == "lqf":
                    if prev_light == 0:  # NS has green
                        raw_action = 1 if prev_ew > prev_ns and env.time_since_switch >= 3 else 0
                    elif prev_light == 1:  # EW has green
                        raw_action = 1 if prev_ns > prev_ew and env.time_since_switch >= 3 else 0
                    else:
                        raw_action = 0
                elif mode == "fixed":
                    raw_action = 1 if env.steps > 0 and env.steps % 5 == 0 else 0
                else:  # random
                    raw_action = random.choice([0, 1])
                    
                agent_action = "SWITCH LIGHT" if raw_action == 1 else "HOLD LIGHT"
                
            state, reward, done, _ = env.step(raw_action)
            
            if done:
                state = env.reset()
            
            if (prev_light == 0 or prev_light == 3) and prev_ns > 0:
                animating_cars.append(AnimatedCar(WIDTH//2 + 12, HEIGHT//2 - 85, 0, 8, CAR_NS_COLOR, True))
            elif (prev_light == 1 or prev_light == 2) and prev_ew > 0:
                animating_cars.append(AnimatedCar(WIDTH//2 - 85, HEIGHT//2 + 12, 8, 0, CAR_EW_COLOR, False))
                
            ns, ew, light = state
            last_logic_time = current_time

        for car in animating_cars:
            car.move()
            
        animating_cars = [car for car in animating_cars if -50 < car.x < WIDTH + 50 and -50 < car.y < HEIGHT + 50]
        
        # --- THE FIX: NEW Z-INDEX RENDER ORDER ---
        draw_base_roads(screen)               # 1. Draw the ground
        draw_queued_cars(screen, ns, ew)      # 2. Draw stopped cars
        
        for car in animating_cars:            
            car.draw(screen)                  # 3. Draw moving cars
            
        draw_overhead_lights(screen, light)   # 4. Draw lights ON TOP of everything
        draw_hud(screen, font, ns, ew, agent_action, mode_name) # 5. Draw UI
        
        pygame.display.flip()
        clock.tick(RENDER_FPS)

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Traffic Signal Control Visualizer")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="q_learning", 
        choices=["q_learning", "sarsa", "dqn", "lqf", "fixed", "random"],
        help="Control mode: q_learning, sarsa, dqn, lqf, fixed, random (default: q_learning)"
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
            print("Trained Q-table file not found. Training a Q-Learning agent first (500 episodes)...")
            from main import train_agent
            train_agent(env, agent, num_episodes=500)
    elif args.mode == "sarsa":
        agent = SARSAAgent(max_queue=5)
        if os.path.exists("q_table_sarsa.npy"):
            print("Loading trained Q-table for SARSA...")
            agent.q_table = np.load("q_table_sarsa.npy")
        else:
            print("Trained Q-table file not found. Training a SARSA agent first (500 episodes)...")
            from compare_algorithms import train_sarsa
            train_sarsa(env, agent, num_episodes=500)
    elif args.mode == "dqn":
        from dqn_agent import DQNAgent
        agent = DQNAgent(state_dim=6, action_dim=2)
        if os.path.exists("dqn_model.pth"):
            print("Loading trained weights for DQN...")
            agent.load("dqn_model.pth")
        else:
            print("Trained DQN model file not found. Training a DQN agent first (500 episodes)...")
            from compare_algorithms import train_dqn
            train_dqn(env, agent, num_episodes=500)
            
    print(f"\nLaunching visualizer in mode: {args.mode.upper()}")
    run_visualization(agent, env, args.mode)