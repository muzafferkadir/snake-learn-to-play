# Gerekli kütüphanelerin import edilmesi
import pygame  # Oyun arayüzü için
import random  # Rastgele sayı üretimi için
import numpy as np  # Numerik işlemler için
from direction import Direction  # Yön enumları
from constants import *  # Oyun sabitleri

# Oyunda kullanılacak renkler (RGB formatında)
BLACK = (0, 0, 0)  # Arka plan rengi
WHITE = (255, 255, 255)  # Yazı ve çerçeve rengi
GREEN = (50, 205, 50)  # Yılan rengi
RED = (255, 0, 0)  # Elma rengi
GRID_COLOR = (40, 40, 40)  # Izgara çizgileri rengi
BUTTON_COLOR = (70, 130, 180)  # Buton normal rengi (Steel Blue)
BUTTON_HOVER_COLOR = (100, 149, 237)  # Buton hover rengi (Cornflower Blue)
HEADER_COLOR = (25, 25, 25)  # Başlık arkaplan rengi (Koyu gri)

# Oyun parametreleri ve sabitleri
CELL_SIZE = 20  # Her hücrenin boyutu (piksel)
GRID_SIZE = 32  # Izgara boyutu (32x32)
HEADER_HEIGHT = 60  # Üst bilgi çubuğu yüksekliği
SCREEN_WIDTH = CELL_SIZE * GRID_SIZE  # Ekran genişliği
SCREEN_HEIGHT = CELL_SIZE * GRID_SIZE + HEADER_HEIGHT  # Ekran yüksekliği
INITIAL_SPEED = 5  # Başlangıç oyun hızı
SPEED_INCREASE = 1  # Her artışta eklenecek hız miktarı
MAX_SPEED = 15  # Maksimum oyun hızı

class Button:
    """Oyun arayüzünde kullanılan butonların sınıfı
    
    Bu sınıf, oyundaki tüm butonların (hız kontrolü, tekrar oyna vb.)
    görünümünü ve davranışını yönetir.
    """
    def __init__(self, x, y, width, height, text):
        """Buton nesnesinin başlatılması
        
        Args:
            x (int): Butonun x koordinatı
            y (int): Butonun y koordinatı
            width (int): Buton genişliği
            height (int): Buton yüksekliği
            text (str): Buton üzerindeki yazı
        """
        self.rect = pygame.Rect(x, y, width, height)  # Butonun dikdörtgen alanı
        self.text = text  # Buton yazısı
        self.is_hovered = False  # Fare üzerinde mi?
        self.font = pygame.font.Font(None, 24)  # Yazı tipi ve boyutu

    def draw(self, screen):
        """Butonu ekrana çizer
        
        Args:
            screen: Pygame ekran nesnesi
        """
        # Buton rengini belirle (fare üzerindeyse farklı renk)
        color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
        
        # Butonu çiz
        pygame.draw.rect(screen, color, self.rect)  # Buton arkaplanı
        pygame.draw.rect(screen, WHITE, self.rect, 2)  # Buton çerçevesi
        
        # Buton yazısını çiz
        text_surface = self.font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        """Buton olaylarını işler (fare hareketi ve tıklama)
        
        Args:
            event: Pygame olay nesnesi
            
        Returns:
            bool: Tıklandıysa True, değilse False
        """
        # Fare hareketi kontrolü
        if event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.mouse.get_pos()
            self.is_hovered = self.rect.collidepoint(mouse_pos)
        # Fare tıklaması kontrolü (sadece sol tık)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse_pos):
                return True
        return False

class SnakeGame:
    """Yılan oyununun ana sınıfı
    
    Bu sınıf, oyunun tüm mekaniğini yönetir:
    - Oyun döngüsü
    - Grafik arayüzü
    - Yılanın hareketi ve kontrolleri
    - Çarpışma tespiti
    - Skor takibi
    - Eğitim modu özellikleri
    """
    def __init__(self):
        """Oyun nesnesinin başlatılması
        
        Bu fonksiyon:
        1. Pygame'i başlatır
        2. Ekranı ayarlar
        3. Oyun parametrelerini tanımlar
        4. Butonları oluşturur
        5. Oyunu sıfırlar
        """
        # Pygame'i başlat ve temel ayarları yap
        pygame.init()
        self.show_ui = True  # Arayüz gösterme durumu
        self.is_training = False  # Eğitim modu durumu
        
        # Ekran boyutlarını ayarla
        self.info_panel_width = 300  # Eğitim paneli genişliği
        total_width = SCREEN_WIDTH + (self.info_panel_width if self.is_training else 0)
        self.screen = pygame.display.set_mode((total_width, SCREEN_HEIGHT))
        pygame.display.set_caption('Snake Game with AI')
        
        # Oyun zamanlayıcısı ve hız ayarları
        self.clock = pygame.time.Clock()
        self.base_speed = INITIAL_SPEED  # Temel oyun hızı
        self.speed = INITIAL_SPEED  # Güncel oyun hızı
        self.training_speed = INITIAL_SPEED  # Eğitim modu hızı
        self.speed_multiplier = 1  # Hız çarpanı
        
        # Yılanın yön bilgisi
        self.last_direction = Direction.RIGHT  # Başlangıç yönü
        
        # Arayüz butonlarını oluştur
        # 1. Tekrar oyna butonu
        button_width = 200
        button_height = 50
        button_x = (SCREEN_WIDTH - button_width) // 2
        button_y = (SCREEN_HEIGHT - button_height) // 2 + 50
        self.replay_button = Button(
            button_x, button_y, 
            button_width, button_height, 
            "Tekrar Oyna"
        )
        
        # 2. Hız kontrol butonları
        self.speed_buttons = []
        self.speeds = [1, 8, 16, 128, 512]  # Hız seçenekleri
        button_width = 45
        button_height = 25
        spacing = 8  # Butonlar arası boşluk
        
        # Butonların toplam genişliğini hesapla
        total_width = len(self.speeds) * (button_width + spacing) - spacing
        start_x = SCREEN_WIDTH + (self.info_panel_width - total_width) // 2
        
        # Hız butonlarını oluştur
        for i, speed in enumerate(self.speeds):
            x = start_x + i * (button_width + spacing)
            y = SCREEN_HEIGHT - 50
            self.speed_buttons.append(
                Button(x, y, button_width, button_height, f"{speed}x")
            )
        
        # Oyunu başlangıç durumuna getir
        self.reset_game()

    def enable_ui(self):
        """Oyun arayüzünü etkinleştirir
        
        Bu fonksiyon:
        1. Arayüz gösterme bayrağını aktif eder
        2. Eğer Pygame başlatılmamışsa başlatır
        3. Ekranı yeniden ayarlar
        """
        self.show_ui = True  # Arayüzü aktif et
        
        # Pygame başlatılmamışsa başlat
        if not pygame.display.get_init():
            pygame.init()
            # Eğitim paneli ile birlikte ekranı oluştur
            self.screen = pygame.display.set_mode(
                (SCREEN_WIDTH + self.info_panel_width, SCREEN_HEIGHT)
            )

    def disable_ui(self):
        """Oyun arayüzünü devre dışı bırakır
        
        Bu fonksiyon:
        1. Arayüz gösterme bayrağını devre dışı bırakır
        2. Eğer Pygame çalışıyorsa kapatır
        
        Not: Eğitim sırasında performansı artırmak için kullanılır
        """
        self.show_ui = False  # Arayüzü devre dışı bırak
        
        # Pygame çalışıyorsa kapat
        if pygame.display.get_init():
            pygame.display.quit()

    def reset_game(self):
        """Oyunu başlangıç durumuna getirir
        
        Bu fonksiyon:
        1. Yılanı başlangıç konumuna yerleştirir
        2. Yeni bir elma yerleştirir
        3. Skoru sıfırlar
        4. Hızı ayarlar
        5. Oyun durumunu sıfırlar
        """
        # Yılanın başlangıç yönü ve konumu
        self.direction = Direction.RIGHT
        self.snake = [
            (GRID_SIZE//2, GRID_SIZE//2),      # Baş
            (GRID_SIZE//2 - 1, GRID_SIZE//2)   # Gövde
        ]
        
        # Yeni elma yerleştir ve skoru sıfırla
        self.place_apple()
        self.score = 0
        
        # Eğitim modunda değilse hızı sıfırla
        if not self.is_training:
            self.speed = self.base_speed
            
        # Oyun durumunu sıfırla
        self.game_over = False
        self.death_cause = None  # Ölüm nedenini sıfırla

    def place_apple(self):
        """Oyun alanına rastgele bir konumda elma yerleştirir
        
        Not: Elma yılanın üzerine gelmeyecek şekilde yerleştirilir
        """
        while True:
            # Rastgele bir konum seç
            self.apple = (
                random.randint(0, GRID_SIZE-1),  # x koordinatı
                random.randint(0, GRID_SIZE-1)   # y koordinatı
            )
            # Eğer elma yılanın üzerinde değilse döngüyü bitir
            if self.apple not in self.snake:
                break

    def set_training_mode(self, is_training):
        """Eğitim modunu ayarlar
        
        Eğitim modunda:
        1. Hız çarpanına göre hız ayarlanır
        2. Sağ panelde eğitim bilgileri gösterilir
        
        Args:
            is_training (bool): Eğitim modu aktif/pasif
        """
        self.is_training = is_training  # Eğitim modunu ayarla
        
        if is_training:
            # Eğitim hızını ayarla
            self.training_speed = self.base_speed * self.speed_multiplier
            # Eğitim panelini ekle
            self.screen = pygame.display.set_mode(
                (SCREEN_WIDTH + self.info_panel_width, SCREEN_HEIGHT)
            )
        else:
            # Normal moda geç
            self.speed = self.base_speed
            # Eğitim panelini kaldır
            self.screen = pygame.display.set_mode(
                (SCREEN_WIDTH, SCREEN_HEIGHT)
            )

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