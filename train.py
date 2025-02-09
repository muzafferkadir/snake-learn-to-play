# Gerekli kütüphanelerin import edilmesi
import pygame  # Oyun arayüzü için
import numpy as np  # Numerik işlemler için
from game import SnakeGame  # Yılan oyunu sınıfı
from ai_model import SnakeAI  # Yapay zeka modeli
from direction import Direction  # Yön enumları
from constants import INITIAL_SPEED, GRID_SIZE  # Oyun sabitleri
import os  # Dosya işlemleri için
import time  # Zaman ölçümleri için
from collections import deque  # Sabit boyutlu kuyruk yapısı için

class TrainLogger:
    """Eğitim sürecini izlemek ve kayıt tutmak için kullanılan sınıf
    
    Bu sınıf, eğitim sürecinde elde edilen skor, epsilon, kayıp gibi 
    değerleri ve çeşitli istatistikleri tutar ve günceller.
    """
    def __init__(self, log_size=100):
        """TrainLogger sınıfının başlatılması
        
        Args:
            log_size (int): Skor geçmişi için maksimum kayıt sayısı (default: 100)
        """
        # Temel metrikler
        self.scores = deque(maxlen=log_size)  # Son N oyunun skorları
        self.mean_scores = []  # Ortalama skorların geçmişi
        self.max_score = 0  # En yüksek skor
        self.total_games = 0  # Toplam oyun sayısı
        self.epsilon_history = []  # Epsilon değerlerinin geçmişi
        self.loss_history = []  # Kayıp değerlerinin geçmişi
        
        # Ölüm ve performans istatistikleri
        self.wall_deaths = 0  # Duvara çarpma sayısı
        self.self_deaths = 0  # Kendine çarpma sayısı
        self.total_apples = 0  # Toplanan toplam elma sayısı
        self.longest_snake = 0  # En uzun yılan uzunluğu
        
    def update(self, score, epsilon, loss):
        """Eğitim metriklerini günceller
        
        Args:
            score (int): Mevcut oyunun skoru
            epsilon (float): Güncel epsilon değeri
            loss (float): Güncel kayıp değeri (None olabilir)
            
        Returns:
            float: Güncel ortalama skor
        """
        # Temel metrikleri güncelle
        self.scores.append(score)  # Yeni skoru ekle
        self.max_score = max(self.max_score, score)  # Rekor güncelleme
        self.total_games += 1  # Oyun sayısını artır
        self.epsilon_history.append(epsilon)  # Epsilon geçmişini güncelle
        
        # Kayıp değeri varsa kaydet
        if loss is not None:
            self.loss_history.append(loss)
        
        # Ortalama skoru hesapla ve kaydet
        mean_score = np.mean(list(self.scores))
        self.mean_scores.append(mean_score)
        
        return mean_score

def train():
    """Yılanın eğitim sürecini yöneten ana fonksiyon
    
    Bu fonksiyon:
    1. Eğitim parametrelerini ayarlar
    2. Oyun ve AI modelini başlatır
    3. Eğitim döngüsünü yönetir
    4. Modeli belirli aralıklarla kaydeder
    5. İstatistikleri tutar ve gösterir
    """
    # Eğitim parametreleri
    n_games = 1000  # Toplam eğitim oyunu sayısı
    batch_size = 32  # Mini-batch boyutu
    record = 0  # Rekor skor
    
    # Gerekli nesneleri oluştur
    ai = SnakeAI()  # Yapay zeka modeli
    game = SnakeGame()  # Oyun motoru
    logger = TrainLogger()  # Eğitim loglayıcı
    
    # Model kayıt klasörünü oluştur
    if not os.path.exists('models'):
        os.makedirs('models')

    print('Eğitim başlıyor...')
    
    # Eğitim modunu aktifleştir
    game.set_training_mode(True)
    game.training_speed = INITIAL_SPEED * 8  # Başlangıç eğitim hızı 8x
    
    for i in range(n_games):
        # Her 50 oyunda bir UI göster
        show_ui = (i % 50 == 0)
        if not show_ui:
            game.disable_ui()  # UI'ı kapat
        else:
            game.enable_ui()   # UI'ı aç
            print(f'\nOyun: {i+1} başlıyor...')
        
        # Oyunu sıfırla
        game.reset_game()
        game_over = False
        steps = 0
        last_score = 0
        start_time = time.time()
        
        # Başlangıç durumu
        state = ai.get_state(game)
        
        while not game_over:
            if show_ui:
                # Event handling (sadece UI açıkken)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    # Hız kontrol butonları için event handling
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        for i, button in enumerate(game.speed_buttons):
                            if button.handle_event(event):
                                game.update_speed(game.speeds[i])
                    elif event.type == pygame.MOUSEMOTION:
                        for button in game.speed_buttons:
                            button.handle_event(event)

            # AI'nin hareketi seç
            action = ai.act(state)
            
            # Hareketi uygula
            reward, game_over, score, death_cause = play_step(game, action)
            
            # Yeni durumu al
            next_state = ai.get_state(game)
            
            # Hafızaya kaydet
            ai.remember(state, action, reward, next_state, game_over)
            
            # Durumu güncelle
            state = next_state
            
            # Öğrenme
            loss = ai.replay(batch_size)
            
            steps += 1
            
            # Her 100 adımda bir target modeli güncelle
            if steps % 100 == 0:
                ai.update_target_model()
            
            if show_ui:
                # Oyunu çiz (sadece UI açıkken)
                game.draw()
                
                # Loglama bilgilerini çiz
                mean_score = logger.update(score, ai.epsilon, loss)
                game.draw_training_info(
                    game=i + 1,
                    score=score,
                    record=record,
                    mean_score=mean_score,
                    epsilon=ai.epsilon,
                    steps=steps,
                    fps=game.training_speed
                )
                
                pygame.display.flip()
                game.clock.tick(game.training_speed)
        
        # İstatistikleri güncelle
        if death_cause == "DUVAR":
            logger.wall_deaths += 1
        elif death_cause == "KUYRUK":
            logger.self_deaths += 1
            
        logger.total_apples += score
        logger.longest_snake = max(logger.longest_snake, len(game.snake))
        
        # Sadece UI açıkken istatistikleri yazdır
        if show_ui:
            duration = time.time() - start_time
            print(f'Oyun Sonucu:')
            print(f'Skor: {score}, Epsilon: {ai.epsilon:.3f}, Mean Score: {mean_score:.2f}')
            print(f'Steps: {steps}, Süre: {duration:.1f}s, Hız: {game.speed_multiplier}x')
            print(f'Ölüm Nedeni: {death_cause}, Yılan Uzunluğu: {len(game.snake)}')
            print(f'Toplam İstatistikler:')
            print(f'Duvar: {logger.wall_deaths}, Kuyruk: {logger.self_deaths}')
            print(f'Toplam Elma: {logger.total_apples}, En Uzun Yılan: {logger.longest_snake}')
            print('-' * 80)
        
        # Rekor kontrolü (sadece UI açıkken bildir)
        if score > record:
            record = score
            ai.save(f'models/model_record_{record}.pth')
            if show_ui:
                print(f'Yeni Rekor! Skor: {record}')
        
        # Her 100 oyunda bir model kaydet (sadece UI açıkken bildir)
        if (i + 1) % 100 == 0:
            ai.save(f'models/model_checkpoint_{i+1}.pth')
            if show_ui:
                print(f'Model kaydedildi: Checkpoint {i+1}')

def play_step(game, action):
    """Oyunda bir adım ilerlemeyi sağlayan fonksiyon
    
    Bu fonksiyon:
    1. AI'nin seçtiği aksiyonu yöne çevirir
    2. Yılanı hareket ettirir
    3. Ödül hesaplaması yapar
    
    Args:
        game (SnakeGame): Oyun nesnesi
        action (int): AI'nin seçtiği aksiyon (0: düz, 1: sağ, 2: sol)
        
    Returns:
        tuple: (ödül, oyun_bitti_mi, skor, ölüm_nedeni)
    """
    # Ödül başlangıcı
    reward = 0
    
    # Aksiyonu yöne çevir
    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]  # Saat yönünde yönler
    idx = clock_wise.index(game.direction)  # Mevcut yönün indeksi
    
    # Yeni yönü belirle
    if action == 0:  # Düz git
        new_dir = clock_wise[idx]
    elif action == 1:  # Sağa dön
        new_dir = clock_wise[(idx + 1) % 4]
    else:  # Sola dön
        new_dir = clock_wise[(idx - 1) % 4]
    
    # Yeni yönü uygula
    game.direction = new_dir
    
    # Hareket öncesi durumu kaydet
    old_score = game.score  # Mevcut skor
    old_distance = ((game.snake[0][0] - game.apple[0])**2 + 
                   (game.snake[0][1] - game.apple[1])**2)**0.5  # Elmaya olan mesafe
    old_head = game.snake[0]  # Yılanın başı
    
    # Yılanı hareket ettir
    death_cause = game.move_snake()
    
    # Hareket sonrası durumu değerlendir
    if not game.game_over:
        # Yeni baş pozisyonu
        head = game.snake[0]
        
        # Yeni elmaya olan mesafe
        new_distance = ((head[0] - game.apple[0])**2 + 
                       (head[1] - game.apple[1])**2)**0.5
        
        # Her adım için küçük negatif ödül
        # Bu, yılanın gereksiz dönüşler yapmak yerine
        # elmaya doğru hareket etmesini teşvik eder
        reward = -0.1
        
        # Elmaya yaklaşma/uzaklaşma kontrolü
        # Elmaya yaklaşma durumunu kontrol et
        if new_distance < old_distance:
            reward = 0  # Elmaya yaklaşıyorsa ceza verme (nötr durum)
        
        # Yerinde sayma kontrolü
        # Yılanın aynı noktada kalmasını engellemek için
        # bu durumu cezalandırıyoruz
        if (old_head[0], old_head[1]) == (head[0], head[1]):
            reward = -1  # Aynı yerde kalma cezası
    
    # Ölüm durumu kontrolü
    if game.game_over:
        reward = -10  # Ölüm durumunda büyük ceza
        return reward, True, game.score, death_cause
    
    # Elma yeme kontrolü
    if game.score > old_score:
        reward = 20  # Elma yeme durumunda büyük ödül
    
    # Durumun sonuçlarını döndür:
    # - reward: Hesaplanan ödül/ceza değeri
    # - game.game_over: Oyun bitti mi?
    # - game.score: Güncel skor
    # - death_cause: Eğer öldüyse ölüm nedeni
    return reward, game.game_over, game.score, death_cause

if __name__ == '__main__':
    # Program doğrudan çalıştırıldığında eğitimi başlat
    train() 