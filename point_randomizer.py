import pygame
import random
import time

# Initialize pygame
pygame.init()

# Set up display
width, height = 600, 400  # Grid size
win = pygame.display.set_mode((width, height))
pygame.display.set_caption("Random Moving Point")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
POINT_COLOR = (255, 0, 0)  # Red point

# Set the size of the point (for visibility)
point_size = 5

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Main loop control
running = True

# Function to draw the point
def draw_point(x, y):
    pygame.draw.circle(win, POINT_COLOR, (x, y), point_size)

# Main loop
while running:
    # Fill the background
    win.fill(BLACK)

    # Generate random coordinates for the point
    point_x = random.randint(0, width - 1)
    point_y = random.randint(0, height - 1)

    # Draw the point at the random position
    draw_point(point_x, point_y)

    # Update the display
    pygame.display.update()

    # Wait for 1 second before generating a new random position
    time.sleep(1)

    # Check for quit events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Control the frame rate (not needed much here, but useful)
    clock.tick(60)

# Quit pygame
pygame.quit()
