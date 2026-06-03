import pygame
import sys
import numpy as np
from environment import TrafficEnv
from agent import QLearningAgent
from main import train_agent

# Layout Constants
WIDTH, HEIGHT = 800, 600
RENDER_FPS = 60
LOGIC_DELAY_MS = 600

# Refined Color Palette
GRASS_GREEN = (46, 125, 50)
ROAD_GRAY = (45, 45, 45)
LINE_YELLOW = (255, 193, 7)
LINE_WHITE = (245, 245, 245)
RED_LIGHT = (229, 57, 53)
GREEN_LIGHT = (76, 175, 80)
BULB_OFF = (30, 30, 30)
HOUSING_BLACK = (20, 20, 20)
CAR_NS_COLOR = (21, 101, 192)  
CAR_EW_COLOR = (198, 40, 40)   
GLASS_COLOR = (179, 229, 252)  
TEXT_WHITE = (255, 255, 255)

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
        if self.is_ns:
            pygame.draw.rect(screen, self.color, (self.x, self.y, 30, 26), border_radius=5)
            pygame.draw.rect(screen, GLASS_COLOR, (self.x + 4, self.y + 18, 22, 5), border_radius=1)
        else:
            pygame.draw.rect(screen, self.color, (self.x, self.y, 26, 30), border_radius=5)
            pygame.draw.rect(screen, GLASS_COLOR, (self.x + 18, self.y + 4, 5, 22), border_radius=1)


def draw_base_roads(screen):
    """Draws ONLY the ground layer: grass, asphalt, and painted lines."""
    screen.fill(GRASS_GREEN)
    
    pygame.draw.rect(screen, ROAD_GRAY, (WIDTH//2 - 50, 0, 100, HEIGHT))  
    pygame.draw.rect(screen, ROAD_GRAY, (0, HEIGHT//2 - 50, WIDTH, 100))  
    
    # Stop Lines
    pygame.draw.rect(screen, LINE_WHITE, (WIDTH//2, HEIGHT//2 - 53, 50, 4))
    pygame.draw.rect(screen, LINE_WHITE, (WIDTH//2 - 50, HEIGHT//2 + 49, 50, 4))
    pygame.draw.rect(screen, LINE_WHITE, (WIDTH//2 - 53, HEIGHT//2, 4, 50))
    pygame.draw.rect(screen, LINE_WHITE, (WIDTH//2 + 49, HEIGHT//2 - 50, 4, 50))

    # Center Lane Dividers
    for y in range(0, HEIGHT, 30):
        if y < HEIGHT//2 - 50 or y > HEIGHT//2 + 50:
            pygame.draw.rect(screen, LINE_YELLOW, (WIDTH//2 - 1, y + 5, 2, 15))
    for x in range(0, WIDTH, 30):
        if x < WIDTH//2 - 50 or x > WIDTH//2 + 50:
            pygame.draw.rect(screen, LINE_YELLOW, (x + 5, HEIGHT//2 - 1, 15, 2))


def draw_overhead_lights(screen, light_state):
    """Draws the overhead gantries AFTER the cars so they appear on top."""
    ns_is_green = (light_state == 0)
    
    # North Incoming Lane
    pygame.draw.rect(screen, HOUSING_BLACK, (WIDTH//2 + 5, HEIGHT//2 - 70, 40, 16), border_radius=4)
    draw_horizontal_bulbs(screen, WIDTH//2 + 9, HEIGHT//2 - 70, ns_is_green)
    
    # South Incoming Lane
    pygame.draw.rect(screen, HOUSING_BLACK, (WIDTH//2 - 45, HEIGHT//2 + 55, 40, 16), border_radius=4)
    draw_horizontal_bulbs(screen, WIDTH//2 - 41, HEIGHT//2 + 55, ns_is_green)
    
    # West Incoming Lane
    pygame.draw.rect(screen, HOUSING_BLACK, (WIDTH//2 - 70, HEIGHT//2 + 5, 16, 40), border_radius=4)
    draw_vertical_bulbs(screen, WIDTH//2 - 70, HEIGHT//2 + 9, not ns_is_green)
    
    # East Incoming Lane
    pygame.draw.rect(screen, HOUSING_BLACK, (WIDTH//2 + 55, HEIGHT//2 - 45, 16, 40), border_radius=4)
    draw_vertical_bulbs(screen, WIDTH//2 + 55, HEIGHT//2 - 41, not ns_is_green)


def draw_horizontal_bulbs(screen, x, y, is_green):
    pygame.draw.circle(screen, RED_LIGHT if not is_green else BULB_OFF, (x + 8, y + 8), 5)
    pygame.draw.circle(screen, GREEN_LIGHT if is_green else BULB_OFF, (x + 24, y + 8), 5)


def draw_vertical_bulbs(screen, x, y, is_green):
    pygame.draw.circle(screen, RED_LIGHT if not is_green else BULB_OFF, (x + 8, y + 8), 5)
    pygame.draw.circle(screen, GREEN_LIGHT if is_green else BULB_OFF, (x + 8, y + 24), 5)


def draw_queued_cars(screen, ns_queue, ew_queue):
    for i in range(ns_queue):
        x = WIDTH//2 + 10
        y = HEIGHT//2 - 85 - (i * 38)
        pygame.draw.rect(screen, CAR_NS_COLOR, (x, y, 30, 26), border_radius=5)
        pygame.draw.rect(screen, GLASS_COLOR, (x + 4, y + 18, 22, 5), border_radius=1)
        
    for i in range(ew_queue):
        x = WIDTH//2 - 85 - (i * 38)
        y = HEIGHT//2 + 10
        pygame.draw.rect(screen, CAR_EW_COLOR, (x, y, 26, 30), border_radius=5)
        pygame.draw.rect(screen, GLASS_COLOR, (x + 18, y + 4, 5, 22), border_radius=1)


def draw_hud(screen, font, ns, ew, agent_action):
    hud_surface = pygame.Surface((340, 110), pygame.SRCALPHA)
    hud_surface.fill((20, 20, 20, 220)) 
    screen.blit(hud_surface, (20, 20))
    pygame.draw.rect(screen, LINE_WHITE, (20, 20, 340, 110), width=1, border_radius=4)
    
    title_font = pygame.font.SysFont(None, 24, bold=True)
    screen.blit(title_font.render("RL SIMULATION METRICS", True, LINE_YELLOW), (35, 30))
    screen.blit(font.render(f"NS Lane Queue: {ns}/5  |  EW Lane Queue: {ew}/5", True, TEXT_WHITE), (35, 60))
    screen.blit(font.render(f"AI Selected Choice: {agent_action}", True, TEXT_WHITE), (35, 90))


def run_visualization(agent, env):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Reinforcement Learning Traffic Signal Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
    
    state = env.reset()
    ns, ew, light = state
    agent_action = "INITIALIZING..."
    
    animating_cars = []
    last_logic_time = pygame.time.get_ticks()
    
    while True:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        if current_time - last_logic_time > LOGIC_DELAY_MS:
            prev_ns, prev_ew, prev_light = state
            
            raw_action = agent.choose_action(state, epsilon=0.0)
            agent_action = "SWITCH LIGHT PHASE" if raw_action == 1 else "HOLD LIGHT PHASE"
            state, reward, done, _ = env.step(raw_action)
            
            if done:
                state = env.reset()
            
            if prev_light == 0 and prev_ns > 0:
                animating_cars.append(AnimatedCar(WIDTH//2 + 10, HEIGHT//2 - 85, 0, 8, CAR_NS_COLOR, True))
            elif prev_light == 1 and prev_ew > 0:
                animating_cars.append(AnimatedCar(WIDTH//2 - 85, HEIGHT//2 + 10, 8, 0, CAR_EW_COLOR, False))
                
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
        draw_hud(screen, font, ns, ew, agent_action) # 5. Draw UI
        
        pygame.display.flip()
        clock.tick(RENDER_FPS)

if __name__ == "__main__":
    print("Training a new RL agent for 500 episodes before visualization...")
    train_env = TrafficEnv(max_queue=5, max_steps=100)
    trained_agent = QLearningAgent(max_queue=5)
    train_agent(train_env, trained_agent, num_episodes=500)
    
    print("\nTraining complete! Launching Pygame visualizer...")
    eval_env = TrafficEnv(max_queue=5, max_steps=100)
    run_visualization(trained_agent, eval_env)