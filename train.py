import pygame
import numpy as np
from game import SnakeGame
from ai_model import SnakeAI
from direction import Direction
from constants import INITIAL_SPEED, GRID_SIZE
import os
import time
from collections import deque

class TrainLogger:
    def __init__(self, log_size=100):
        self.scores = deque(maxlen=log_size)
        self.mean_scores = []
        self.max_score = 0
        self.total_games = 0
        self.epsilon_history = []
        self.loss_history = []
        # Yeni istatistikler
        self.wall_deaths = 0
        self.self_deaths = 0
        self.total_apples = 0
        self.longest_snake = 0
        
    def update(self, score, epsilon, loss):
        self.scores.append(score)
        self.max_score = max(self.max_score, score)
        self.total_games += 1
        self.epsilon_history.append(epsilon)
        if loss is not None:
            self.loss_history.append(loss)
        
        mean_score = np.mean(list(self.scores))
        self.mean_scores.append(mean_score)
        
        return mean_score

def train():
    n_games = 1000
    batch_size = 32
    record = 0
    ai = SnakeAI()
    game = SnakeGame()
    logger = TrainLogger()
    
    # Eğitim klasörünü oluştur
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
    # Ödül başlangıcı
    reward = 0
    
    # Yönü güncelle
    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    idx = clock_wise.index(game.direction)
    
    if action == 0:  # Düz
        new_dir = clock_wise[idx]
    elif action == 1:  # Sağa dön
        new_dir = clock_wise[(idx + 1) % 4]
    else:  # Sola dön
        new_dir = clock_wise[(idx - 1) % 4]
    
    game.direction = new_dir
    
    # Skoru kaydet
    old_score = game.score
    old_distance = ((game.snake[0][0] - game.apple[0])**2 + (game.snake[0][1] - game.apple[1])**2)**0.5
    old_head = game.snake[0]
    
    # Hareketi uygula
    death_cause = game.move_snake()
    
    # Yeni mesafeyi hesapla
    if not game.game_over:
        head = game.snake[0]
        new_distance = ((head[0] - game.apple[0])**2 + (head[1] - game.apple[1])**2)**0.5
        
        # Her adım için küçük negatif ödül (elmaya doğru hareket etmeye teşvik)
        reward = -0.1
        
        # Elmaya yaklaşma/uzaklaşma kontrolü
        if new_distance < old_distance:
            reward = 0  # Elmaya yaklaşıyorsa ceza verme
        
        # Döngüye girmeyi engelleme
        if (old_head[0], old_head[1]) == (head[0], head[1]):
            reward = -1  # Aynı yerde kalma cezası
    
    # Ödülü hesapla
    if game.game_over:
        reward = -10  # Ölüm cezası
        return reward, True, game.score, death_cause
    
    # Elma yeme ödülü
    if game.score > old_score:
        reward = 20  # Elma yeme ödülü
    
    return reward, game.game_over, game.score, death_cause

if __name__ == '__main__':
    train() 