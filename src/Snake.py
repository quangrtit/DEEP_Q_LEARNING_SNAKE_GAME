import pygame 
import numpy
import random



BLOCK_SIZE = 10
FRAMESPEED = 500
DIS_WIDTH = 800
DIS_HEIGHT = 600
YELLOW = (255, 255, 102)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)
pygame.init()
class Snake:
    def __init__(self):
        self.reset()
    def reset(self):
        self.clock = pygame.time.Clock() 
        self.display = pygame.display.set_mode((DIS_WIDTH, DIS_HEIGHT))
        pygame.display.set_caption("ANH QUẢNG")
        x_head, y_head = 4 * BLOCK_SIZE, 4 * BLOCK_SIZE
        self.snake = [(x_head, y_head)]
        self.x_food = random.randint(0, DIS_WIDTH - BLOCK_SIZE) // BLOCK_SIZE * BLOCK_SIZE
        self.y_food = random.randint(0, DIS_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE * BLOCK_SIZE
        self.x_change = BLOCK_SIZE
        self.y_change = 0
        self.fps = FRAMESPEED
        self.size = 1 # score = self.size - 1
        self.covert_action = {
            0: "left", 
            1: "right", 
            2: "up",
            3: "down"
        }
        return self.get_state()
    def get_state(self):
        # we have state with 12 status: direction: left, right, up, down, danger: left, right, up, down, food: left, right, up, down
        x_head = self.snake[-1][0] 
        y_head = self.snake[-1][1]
        state = [
            # direction of snake 
            self.x_change == -BLOCK_SIZE,
            self.x_change == BLOCK_SIZE, 
            self.y_change == -BLOCK_SIZE, 
            self.y_change == BLOCK_SIZE,
            # danger
            (x_head, y_head) in self.snake[:-1],
            x_head < 0, 
            x_head >= DIS_WIDTH,
            y_head < 0,
            y_head >= DIS_HEIGHT,
            # food 
            x_head > self.x_food,
            x_head < self.x_food,
            y_head > self.y_food,
            y_head < self.y_food
        ]
        return numpy.array(state, dtype=int)
    def step(self, action):
        next_state = None 
        reward = 0
        done = False
        if self.covert_action[action] == "left":
            if self.x_change != BLOCK_SIZE:
                self.x_change = -BLOCK_SIZE
                self.y_change = 0
        elif self.covert_action[action] == "right":
            if self.x_change != -BLOCK_SIZE:
                self.x_change = BLOCK_SIZE
                self.y_change = 0
        elif self.covert_action[action] == "up":
            if self.y_change != BLOCK_SIZE:
                self.x_change = 0
                self.y_change = -BLOCK_SIZE
        elif self.covert_action[action] == "down":
            if self.y_change != -BLOCK_SIZE:
                self.x_change = 0
                self.y_change = BLOCK_SIZE
        x_head, y_head = self.snake[-1][0] + self.x_change, self.snake[-1][1] + self.y_change
        self.snake.append((x_head, y_head))
        # check snake eats food
        if x_head == self.x_food and y_head == self.y_food:
            reward = 10
            self.x_food = random.randint(0, DIS_WIDTH - BLOCK_SIZE) // BLOCK_SIZE * BLOCK_SIZE
            self.y_food = random.randint(0, DIS_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE * BLOCK_SIZE
            self.size += 1
        else: 
            del self.snake[0]
        # check snake in wall or tail
        if ((x_head, y_head) in self.snake[:-1]) or x_head < 0 or x_head >= DIS_WIDTH or y_head < 0 or y_head >= DIS_HEIGHT: 
            reward = -10
            done = True
        next_state = self.get_state()
        return next_state, reward, done, self.size
    def render(self):
        self.display.fill(BLUE)
        # draw snake 
        for body in self.snake:
            pygame.draw.rect(self.display, YELLOW, (body[0], body[1], BLOCK_SIZE, BLOCK_SIZE))
        # draw food
        pygame.draw.rect(self.display, BLACK, (self.x_food, self.y_food, BLOCK_SIZE, BLOCK_SIZE))
        # draw score
        font = pygame.font.SysFont("comicsansms", 35)
        value = font.render(f"Score: {self.size - 1}", True, GREEN)
        self.display.blit(value, [0, 0])  

        pygame.display.update()
        self.clock.tick(self.fps)
    def act(self):
        return numpy.random.randint(0, 3)
    def close_game(self):
        pygame.quit()
    def quit_game(self):
        pygame.quit()
    def delay_game(self, time=1000):
        pygame.time.delay(time)
    def change_fps(self, fps):
        self.fps = fps
if __name__ == "__main__":
    pass

