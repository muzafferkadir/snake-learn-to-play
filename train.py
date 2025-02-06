import pygame
import numpy as np
from game import SnakeGame
from ai_model import SnakeAI
from direction import Direction
from constants import INITIAL_SPEED
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
    game.training_speed = INITIAL_SPEED * 16  # Başlangıç eğitim hızı 16x
    
    for i in range(n_games):
        # Oyunu sıfırla
        game.reset_game()
        game_over = False
        steps = 0
        last_score = 0
        start_time = time.time()
        
        # Başlangıç durumu
        state = ai.get_state(game)
        
        while not game_over:
            # Event handling
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
            reward, game_over, score = play_step(game, action)
            
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
            
            # Sonsuz döngüyü engelle
            if steps > 1000:  # Maksimum adım sayısı
                game_over = True
                reward = -10  # Ceza ver
            
            # Oyunu çiz
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
        
        # Oyun istatistiklerini yazdır
        duration = time.time() - start_time
        print(f'Oyun: {i+1}, Skor: {score}, Epsilon: {ai.epsilon:.3f}, ' +
              f'Mean Score: {mean_score:.2f}, Steps: {steps}, ' +
              f'Süre: {duration:.1f}s, Hız: {game.speed_multiplier}x')
        
        # Rekor kontrolü
        if score > record:
            record = score
            ai.save(f'models/model_record_{record}.pth')
        
        # Her 100 oyunda bir model kaydet
        if (i + 1) % 100 == 0:
            ai.save(f'models/model_checkpoint_{i+1}.pth')

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
    
    # Hareketi uygula
    game.move_snake()
    
    # Ödülü hesapla
    if game.game_over:
        reward = -10
        return reward, True, game.score
    
    if game.score > old_score:
        reward = 10
    else:
        reward = -0.1  # Her adımda küçük bir ceza
    
    return reward, game.game_over, game.score

if __name__ == '__main__':
    train() 