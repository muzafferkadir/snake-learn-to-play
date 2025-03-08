import pygame
import heapq
import random
from collections import deque
from game import SnakeGame
from direction import Direction
from constants import GRID_SIZE
import argparse

class SnakeAStarAI:
    """A* algoritması kullanan yılan yapay zekası.
    
    Bu sınıf, klasik A* arama algoritmasını kullanarak elmalara en kısa
    yolu bulmayı ve yılanın kendi kuyruğuna çarpmasını engellemeyi amaçlar.
    """
    
    def __init__(self):
        """SnakeAStarAI sınıfının başlatılması"""
        # İstatistikler
        self.path = []  # Hesaplanan mevcut yol
        self.total_moves = 0  # Toplam hamle sayısı
        self.apples_eaten = 0  # Yenen elma sayısı
        self.path_recomputes = 0  # Yol yeniden hesaplama sayısı
        
    def heuristic(self, a, b):
        """İki nokta arası tahmini mesafeyi hesaplar (Manhattan mesafesi)
        
        Args:
            a (tuple): Başlangıç noktası (x, y)
            b (tuple): Hedef noktası (x, y)
            
        Returns:
            int: İki nokta arası tahmini mesafe
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def is_valid_move(self, pos, game):
        """Verilen pozisyonun geçerli olup olmadığını kontrol eder
        
        Args:
            pos (tuple): Kontrol edilecek pozisyon (x, y)
            game (SnakeGame): Oyun nesnesi
            
        Returns:
            bool: Pozisyon geçerliyse True, değilse False
        """
        # Sınırları kontrol et
        if pos[0] < 0 or pos[0] >= GRID_SIZE or pos[1] < 0 or pos[1] >= GRID_SIZE:
            return False
        
        # Yılanın vücuduyla çarpışma kontrolü
        # NOT: Kuyruk parçası ilerleyecekse, çarpışma olmayacak
        if pos in game.snake[:-1]:
            return False
            
        return True
    
    def find_path_to_apple(self, game):
        """A* algoritmasını kullanarak elmaya giden en kısa yolu bulur
        
        Args:
            game (SnakeGame): Oyun nesnesi
            
        Returns:
            list: Elmaya giden yol listesi veya boş liste (yol bulunamadıysa)
        """
        # Başlangıç ve hedef noktaları
        start = game.snake[0]
        goal = game.apple
        
        # A* algoritması için veri yapıları
        open_set = []  # Öncelik kuyruğu
        heapq.heappush(open_set, (0, start, 0))  # (f_score, pozisyon, adım)
        came_from = {}  # Hangi noktadan gelindi
        g_score = {start: 0}  # Başlangıçtan bir noktaya kadar olan maliyet
        f_score = {start: self.heuristic(start, goal)}  # Toplam tahmini maliyet
        
        # Daha sonra yolun doğru sırayla çıkarılması için
        # Her nokta için adım numarasını saklayalım
        step_count = {start: 0}
        
        # A* algoritması
        while open_set:
            # En düşük f skorlu noktayı al
            _, current, steps = heapq.heappop(open_set)
            
            # Hedefe ulaşıldıysa yolu oluştur
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()  # Doğru sırada olması için tersine çevir
                return path
            
            # Dört yöne de bak
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # aşağı, sağ, yukarı, sol
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Geçersiz hareketleri atla
                if not self.is_valid_move(neighbor, game):
                    continue
                
                # Yeni g değerini hesapla
                tentative_g = g_score[current] + 1
                
                # Daha iyi bir yol bulunduysa güncelle
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Bu nokta için nereden geldiğimizi kaydet
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    step_count[neighbor] = steps + 1
                    
                    # Eğer açık kümede değilse ekle
                    if not any(neighbor == pos for _, pos, _ in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor, steps + 1))
        
        # Yol bulunamadıysa güvenli hareket bul
        return self.find_safe_move(game)
    
    def find_safe_move(self, game):
        """Elmaya giden yol bulunamadığında güvenli bir hamle bul
        
        Bu fonksiyon, yılanın hayatta kalmasını sağlayacak güvenli bir hamle
        bulur. Öncelikle en uzak boş alana doğru hareket etmeye çalışır.
        
        Args:
            game (SnakeGame): Oyun nesnesi
            
        Returns:
            list: Güvenli hamlelerden oluşan bir yol
        """
        head = game.snake[0]
        safe_moves = []
        
        # Güvenli hamleleri bul
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (head[0] + dx, head[1] + dy)
            if self.is_valid_move(neighbor, game):
                safe_moves.append(neighbor)
        
        if not safe_moves:
            return []  # Hiç güvenli hamle yoksa boş liste döndür
        
        # Genişlik öncelikli arama ile en uzak noktayı bul
        return self.find_longest_path(game, safe_moves)
    
    def find_longest_path(self, game, safe_moves):
        """Genişlik öncelikli arama ile en uzak boş noktayı bul
        
        Args:
            game (SnakeGame): Oyun nesnesi
            safe_moves (list): Güvenli hamle listesi
            
        Returns:
            list: En uzak noktaya giden yolun başlangıcı
        """
        if not safe_moves:
            return []
            
        # En uzak nokta için en iyi mesafe
        best_distance = -1
        best_move = safe_moves[0]
        
        # Her güvenli hamle için genişlik öncelikli arama yap
        for move in safe_moves:
            # BFS için veri yapıları
            queue = deque([(move, 1)])  # (pozisyon, mesafe)
            visited = {move}
            
            max_distance = 1  # Bu hamle için maksimum mesafe
            
            while queue:
                pos, distance = queue.popleft()
                
                # Maksimum mesafeyi güncelle
                max_distance = max(max_distance, distance)
                
                # Dört yöne bakarak BFS devam et
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    neighbor = (pos[0] + dx, pos[1] + dy)
                    
                    # Geçersiz veya ziyaret edilmiş noktaları atla
                    if not self.is_valid_move(neighbor, game) or neighbor in visited:
                        continue
                    
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
            
            # Bu hamle daha uzağa gidebiliyorsa güncelle
            if max_distance > best_distance:
                best_distance = max_distance
                best_move = move
        
        return [best_move]  # Sadece ilk hamleyi döndür
    
    def get_next_direction(self, game):
        """Bir sonraki hareket yönünü belirler
        
        Args:
            game (SnakeGame): Oyun nesnesi
            
        Returns:
            Direction: Bir sonraki hareket yönü
        """
        head = game.snake[0]
        
        # Yol boşsa veya başka bir yolu takip etmek mümkün değilse,
        # yeni bir yol hesapla
        if not self.path:
            self.path = self.find_path_to_apple(game)
            self.path_recomputes += 1
        
        # Hala yol bulunamadıysa rastgele bir yön seç
        if not self.path:
            directions = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_pos = (head[0] + dx, head[1] + dy)
                if self.is_valid_move(new_pos, game):
                    if dx == 1:
                        directions.append(Direction.RIGHT)
                    elif dx == -1:
                        directions.append(Direction.LEFT)
                    elif dy == 1:
                        directions.append(Direction.DOWN)
                    elif dy == -1:
                        directions.append(Direction.UP)
            
            if not directions:
                # Hiç güvenli yön yoksa mevcut yönde devam et
                return game.direction
            
            return random.choice(directions)
        
        # Yoldaki bir sonraki noktayı al
        next_pos = self.path[0]
        self.path = self.path[1:]  # Yoldan bu noktayı çıkar
        
        # Bir sonraki hareketin yönünü belirle
        dx = next_pos[0] - head[0]
        dy = next_pos[1] - head[1]
        
        if dx == 1:
            return Direction.RIGHT
        elif dx == -1:
            return Direction.LEFT
        elif dy == 1:
            return Direction.DOWN
        elif dy == -1:
            return Direction.UP
        
        # Varsayılan olarak mevcut yönde devam et
        return game.direction
    
    def move(self, game):
        """Yılanı bir sonraki pozisyona hareket ettirir
        
        Args:
            game (SnakeGame): Oyun nesnesi
            
        Returns:
            string: Ölüm nedeni veya None
        """
        # Bir sonraki yönü belirle
        next_direction = self.get_next_direction(game)
        
        # Yönü uygula
        game.direction = next_direction
        
        # Yılanı hareket ettir
        death_cause = game.move_snake()
        
        # İstatistikleri güncelle
        self.total_moves += 1
        old_apples = self.apples_eaten
        self.apples_eaten = game.score
        
        # Eğer elma yenildiyse veya ölündüyse yolu sıfırla
        if self.apples_eaten > old_apples or game.game_over:
            self.path = []
        
        return death_cause

def play_astar_ai():
    """A* yapay zekasıyla yılan oyununu oynatır"""
    game = SnakeGame()
    ai = SnakeAStarAI()
    
    print("A* yapay zekasıyla oyun başlatılıyor...")
    pygame_active = True
    
    try:
        # Oyun döngüsü
        while pygame_active:
            # AI'nın hareketi
            death_cause = ai.move(game)
            
            # Oyunu çiz
            game.draw()
            pygame.display.flip()
            
            # Oyun hızı
            game.clock.tick(game.speed)
            
            # Çıkış kontrolü
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame_active = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame_active = False
                        break
                    # Hızı ayarla
                    elif event.key == pygame.K_1:
                        game.speed = 5  # Normal hız
                    elif event.key == pygame.K_2:
                        game.speed = 10  # 2x hız
                    elif event.key == pygame.K_3:
                        game.speed = 20  # 4x hız
                    elif event.key == pygame.K_4:
                        game.speed = 40  # 8x hız
                    elif event.key == pygame.K_5:
                        game.speed = 60  # 12x hız
            
            # Oyun bitti mi?
            if game.game_over:
                print(f"Oyun bitti! Skor: {game.score}, Ölüm nedeni: {death_cause}")
                print(f"Toplam hamle: {ai.total_moves}, Yol yeniden hesaplama: {ai.path_recomputes}")
                game.reset_game()
                ai.path = []  # Yolu sıfırla
    except Exception as e:
        print(f"Hata oluştu: {e}")
    finally:
        if pygame.display.get_init():
            pygame.quit()
        print("Oyun kapatıldı.")

if __name__ == "__main__":
    play_astar_ai() 