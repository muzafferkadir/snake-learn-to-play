import pygame
import random
import numpy as np
from direction import Direction
from constants import *

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

class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.is_hovered = False
        self.font = pygame.font.Font(None, 24)  # Buton font boyutunu küçülttük

    def draw(self, screen):
        color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 2)  # Border
        
        text_surface = self.font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.mouse.get_pos()
            self.is_hovered = self.rect.collidepoint(mouse_pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Sol tık
            mouse_pos = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse_pos):
                return True
        return False

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.show_ui = True  # UI gösterme kontrolü
        self.is_training = False  # Eğitim modu kontrolü
        # Eğitim paneli için sağda ekstra alan
        self.info_panel_width = 300
        self.screen = pygame.display.set_mode((SCREEN_WIDTH + (self.info_panel_width if self.is_training else 0), SCREEN_HEIGHT))
        pygame.display.set_caption('Snake Game with AI')
        self.clock = pygame.time.Clock()
        self.base_speed = INITIAL_SPEED  # Temel hız
        self.speed = INITIAL_SPEED  # Oyun hızı
        self.training_speed = INITIAL_SPEED  # Eğitim hızı
        self.speed_multiplier = 1
        self.last_direction = Direction.RIGHT  # Son yön bilgisini tutmak için
        
        # Tekrar oyna butonu
        button_width = 200
        button_height = 50
        button_x = (SCREEN_WIDTH - button_width) // 2
        button_y = (SCREEN_HEIGHT - button_height) // 2 + 50
        self.replay_button = Button(button_x, button_y, button_width, button_height, "Tekrar Oyna")
        
        # Hız kontrol butonları
        self.speed_buttons = []
        self.speeds = [1, 8, 16, 128, 512]
        button_width = 45
        button_height = 25
        spacing = 8
        total_width = len(self.speeds) * (button_width + spacing) - spacing
        start_x = SCREEN_WIDTH + (self.info_panel_width - total_width) // 2
        
        for i, speed in enumerate(self.speeds):
            x = start_x + i * (button_width + spacing)
            y = SCREEN_HEIGHT - 50
            self.speed_buttons.append(
                Button(x, y, button_width, button_height, f"{speed}x")
            )
        
        self.reset_game()

    def enable_ui(self):
        """UI'ı etkinleştir"""
        self.show_ui = True
        if not pygame.display.get_init():
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH + self.info_panel_width, SCREEN_HEIGHT))

    def disable_ui(self):
        """UI'ı devre dışı bırak"""
        self.show_ui = False
        if pygame.display.get_init():
            pygame.display.quit()

    def reset_game(self):
        self.direction = Direction.RIGHT
        self.snake = [(GRID_SIZE//2, GRID_SIZE//2), (GRID_SIZE//2 - 1, GRID_SIZE//2)]
        self.place_apple()
        self.score = 0
        if not self.is_training:
            self.speed = self.base_speed
        self.game_over = False
        self.death_cause = None  # Ölüm nedenini sıfırla

    def place_apple(self):
        while True:
            self.apple = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if self.apple not in self.snake:
                break

    def set_training_mode(self, is_training):
        self.is_training = is_training
        if is_training:
            self.training_speed = self.base_speed * self.speed_multiplier
            # Training modunda sağ panel ekle
            self.screen = pygame.display.set_mode((SCREEN_WIDTH + self.info_panel_width, SCREEN_HEIGHT))
        else:
            self.speed = self.base_speed
            # Normal modda sağ panel olmadan
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    def update_speed(self, multiplier):
        old_multiplier = self.speed_multiplier
        self.speed_multiplier = multiplier
        if self.is_training:
            self.training_speed = self.base_speed * multiplier
            print(f"Eğitim hızı güncellendi: {old_multiplier}x -> {multiplier}x (FPS: {self.training_speed})")

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if not self.game_over:
                    # Yılanın anlık yönünü kontrol et ve son yönünü kullan
                    if event.key == pygame.K_UP and self.last_direction != Direction.DOWN:
                        self.direction = Direction.UP
                    if event.key == pygame.K_DOWN and self.last_direction != Direction.UP:
                        self.direction = Direction.DOWN
                    if event.key == pygame.K_LEFT and self.last_direction != Direction.RIGHT:
                        self.direction = Direction.LEFT
                    if event.key == pygame.K_RIGHT and self.last_direction != Direction.LEFT:
                        self.direction = Direction.RIGHT
            
            # Oyun bitti ekranında buton kontrolü
            if self.game_over:
                if self.replay_button.handle_event(event):
                    self.reset_game()
            
            # Hız kontrol butonları kontrolü
            if self.is_training:  # Sadece eğitim modunda hız kontrolü aktif
                for i, button in enumerate(self.speed_buttons):
                    if button.handle_event(event):
                        self.update_speed(self.speeds[i])
                    
        return True

    def move_snake(self):
        if self.game_over:
            return None

        head = self.snake[0]
        # Yılanın son yönünü kaydet
        self.last_direction = self.direction
        
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
            self.death_cause = "DUVAR"
            return self.death_cause

        # Kendine çarpma kontrolü
        if new_head in self.snake:
            self.game_over = True
            self.death_cause = "KUYRUK"
            return self.death_cause

        self.snake.insert(0, new_head)

        # Elma yeme kontrolü
        if new_head == self.apple:
            self.score += 1
            if not self.is_training:  # Eğitim modunda değilse hız artışı
                if self.score % 5 == 0:  # Her 5 elmada bir hız artışı
                    # Maksimum hız kontrolü
                    if self.speed < MAX_SPEED:
                        self.speed += SPEED_INCREASE
            self.place_apple()
        else:
            self.snake.pop()

        return None

    def draw(self):
        if not self.show_ui:
            return
            
        self.screen.fill(BLACK)
        
        # Header çizimi
        pygame.draw.rect(self.screen, HEADER_COLOR, (0, 0, SCREEN_WIDTH, HEADER_HEIGHT))
        
        # Skor ve hız gösterimi
        font = pygame.font.Font(None, 36)
        
        # Skor
        score_text = font.render(f'Skor: {self.score}', True, WHITE)
        score_rect = score_text.get_rect(midleft=(20, HEADER_HEIGHT//2))
        self.screen.blit(score_text, score_rect)
        
        # Hız (sadece training modunda değilse göster)
        if not self.is_training:
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

    def draw_training_info(self, game, score, record, mean_score, epsilon, steps, fps):
        if not self.show_ui or not self.is_training:
            return
            
        # Sağ panel arkaplanı
        panel_x = SCREEN_WIDTH
        pygame.draw.rect(self.screen, HEADER_COLOR, 
                        (panel_x, 0, self.info_panel_width, SCREEN_HEIGHT))
        
        # Başlık
        font_large = pygame.font.Font(None, 36)
        title = font_large.render('Eğitim Bilgileri', True, WHITE)
        title_rect = title.get_rect(centerx=panel_x + self.info_panel_width//2, y=20)
        self.screen.blit(title, title_rect)
        
        # Alt çizgi
        pygame.draw.line(self.screen, WHITE, 
                        (panel_x + 20, 50), 
                        (panel_x + self.info_panel_width - 20, 50))
        
        font = pygame.font.Font(None, 28)
        y = 70
        spacing = 35
        
        # Temel bilgiler
        infos = [
            ('Oyun', f'{game}'),
            ('Skor', f'{score}'),
            ('Rekor', f'{record}'),
            ('Ort. Skor', f'{mean_score:.2f}'),
            ('Epsilon', f'{epsilon:.3f}'),
            ('Adımlar', f'{steps}'),
            ('FPS', f'{self.training_speed}'),
            ('Yılan Uzunluğu', f'{len(self.snake)}'),
            ('Hız Çarpanı', f'{self.speed_multiplier}x')
        ]
        
        for label, value in infos:
            text = font.render(f'{label}:', True, WHITE)
            value_text = font.render(str(value), True, WHITE)
            
            # Label sol tarafta
            self.screen.blit(text, (panel_x + 20, y))
            # Değer sağ tarafta
            value_rect = value_text.get_rect(
                right=panel_x + self.info_panel_width - 20, 
                top=y
            )
            self.screen.blit(value_text, value_rect)
            
            y += spacing
        
        # Aktif hız butonunu vurgula
        for i, button in enumerate(self.speed_buttons):
            is_active = self.speed_multiplier == self.speeds[i]
            if is_active:
                # Aktif buton için ekstra vurgu
                pygame.draw.rect(self.screen, WHITE, button.rect, 3)
            button.draw(self.screen)

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
            pygame.display.flip()  # Ekranı güncelle

if __name__ == "__main__":
    game = SnakeGame()
    game.run()
    pygame.quit() 