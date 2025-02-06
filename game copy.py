import pygame
import random
import numpy as np
from enum import Enum

# Renkler
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (50, 205, 50)
RED = (255, 0, 0)
GRID_COLOR = (40, 40, 40)
BUTTON_COLOR = (70, 130, 180)  # Steel Blue
BUTTON_HOVER_COLOR = (100, 149, 237)  # Cornflower Blue
HEADER_COLOR = (25, 25, 25)  # Koyu gri

# Oyun sabitleri
CELL_SIZE = 20
GRID_SIZE = 32
HEADER_HEIGHT = 60
SCREEN_WIDTH = CELL_SIZE * GRID_SIZE
SCREEN_HEIGHT = CELL_SIZE * GRID_SIZE + HEADER_HEIGHT
INITIAL_SPEED = 5  # Başlangıç hızı
SPEED_INCREASE = 1  # Hız artış miktarı
MAX_SPEED = 15  # Maksimum hız sınırı

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.is_hovered = False
        self.font = pygame.font.Font(None, 36)

    def draw(self, screen):
        color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 2)  # Border
        
        text_surface = self.font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                return True
        return False

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        
        # Tekrar oyna butonu
        button_width = 200
        button_height = 50
        button_x = (SCREEN_WIDTH - button_width) // 2
        button_y = (SCREEN_HEIGHT + 40) // 2  # Header'ı hesaba katarak konumu ayarladık
        self.replay_button = Button(button_x, button_y, button_width, button_height, "Tekrar Oyna")
        
        self.reset_game()

    def reset_game(self):
        self.direction = Direction.RIGHT
        self.snake = [(GRID_SIZE//2, GRID_SIZE//2), (GRID_SIZE//2 - 1, GRID_SIZE//2)]
        self.place_apple()
        self.score = 0
        self.speed = INITIAL_SPEED
        self.game_over = False

    def place_apple(self):
        while True:
            self.apple = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if self.apple not in self.snake:
                break

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if not self.game_over:
                    if event.key == pygame.K_UP and self.direction != Direction.DOWN:
                        self.direction = Direction.UP
                    if event.key == pygame.K_DOWN and self.direction != Direction.UP:
                        self.direction = Direction.DOWN
                    if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                        self.direction = Direction.LEFT
                    if event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                        self.direction = Direction.RIGHT
            
            # Oyun bitti ekranında buton kontrolü
            if self.game_over:
                if self.replay_button.handle_event(event):
                    self.reset_game()
                    
        return True

    def move_snake(self):
        if self.game_over:
            return

        head = self.snake[0]
        if self.direction == Direction.UP:
            new_head = (head[0], head[1] - 1)
        elif self.direction == Direction.DOWN:
            new_head = (head[0], head[1] + 1)
        elif self.direction == Direction.LEFT:
            new_head = (head[0] - 1, head[1])
        else:
            new_head = (head[0] + 1, head[1])

        # Duvar çarpışma kontrolü
        if (new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE):
            self.game_over = True
            return

        # Kendine çarpma kontrolü
        if new_head in self.snake:
            self.game_over = True
            return

        self.snake.insert(0, new_head)

        # Elma yeme kontrolü
        if new_head == self.apple:
            self.score += 1
            if self.score % 5 == 0:  # Her 5 elmada bir hız artışı
                # Maksimum hız kontrolü
                if self.speed < MAX_SPEED:
                    self.speed += SPEED_INCREASE
            self.place_apple()
        else:
            self.snake.pop()

    def draw(self):
        self.screen.fill(BLACK)
        
        # Header çizimi
        pygame.draw.rect(self.screen, HEADER_COLOR, (0, 0, SCREEN_WIDTH, HEADER_HEIGHT))
        
        # Skor ve hız gösterimi
        font = pygame.font.Font(None, 36)
        
        # Skor
        score_text = font.render(f'Skor: {self.score}', True, WHITE)
        score_rect = score_text.get_rect(midleft=(20, HEADER_HEIGHT//2))
        self.screen.blit(score_text, score_rect)
        
        # Hız
        speed_text = font.render(f'Hız: {self.speed}', True, WHITE)
        speed_rect = speed_text.get_rect(midright=(SCREEN_WIDTH - 20, HEADER_HEIGHT//2))
        self.screen.blit(speed_text, speed_rect)
        
        # Izgara çizimi
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, 
                           (x, HEADER_HEIGHT), (x, SCREEN_HEIGHT))
        for y in range(HEADER_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, 
                           (0, y), (SCREEN_WIDTH, y))

        # Yılan çizimi
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN,
                           (segment[0] * CELL_SIZE, 
                            segment[1] * CELL_SIZE + HEADER_HEIGHT,
                            CELL_SIZE-1, CELL_SIZE-1))

        # Elma çizimi
        pygame.draw.rect(self.screen, RED,
                        (self.apple[0] * CELL_SIZE, 
                         self.apple[1] * CELL_SIZE + HEADER_HEIGHT,
                         CELL_SIZE-1, CELL_SIZE-1))

        if self.game_over:
            # Oyun bitti yazısı
            font = pygame.font.Font(None, 74)
            text = font.render('Oyun Bitti!', True, WHITE)
            text_rect = text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            self.screen.blit(text, text_rect)
            
            # Tekrar oyna butonu
            self.replay_button.draw(self.screen)

        pygame.display.flip()

    def get_state(self):
        # AI için oyun durumunu döndür
        state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        # Yılanın konumu
        for x, y in self.snake:
            state[y][x] = 0.5
        
        # Yılan başı
        head_x, head_y = self.snake[0]
        state[head_y][head_x] = 1
        
        # Elmanın konumu
        state[self.apple[1]][self.apple[0]] = -1
        
        return state

    def run(self):
        running = True
        while running:
            running = self.handle_input()
            self.move_snake()
            self.draw()
            self.clock.tick(self.speed)

if __name__ == "__main__":
    game = SnakeGame()
    game.run()
    pygame.quit() 