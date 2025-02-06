import pygame
from game import SnakeGame
from ai_model import SnakeAI
from direction import Direction
import argparse

def play_ai(model_path):
    game = SnakeGame()
    ai = SnakeAI()
    
    # Modeli yükle
    try:
        ai.load(model_path)
        ai.epsilon = 0  # Keşif modunu kapat
        print(f"Model yüklendi: {model_path}")
    except:
        print(f"Model yüklenemedi: {model_path}")
        return

    # Oyun döngüsü
    while True:
        # Durumu al
        state = ai.get_state(game)
        
        # AI'nin hareketi
        action = ai.act(state)
        
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
        
        # Hareketi uygula
        game.move_snake()
        
        # Oyunu çiz
        game.draw()
        pygame.display.flip()
        
        # Oyun hızı
        game.clock.tick(game.speed)
        
        # Çıkış kontrolü
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
        
        # Oyun bitti mi?
        if game.game_over:
            print(f"Oyun bitti! Skor: {game.score}")
            game.reset_game()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Snake AI ile oyna')
    parser.add_argument('--model', type=str, required=True,
                      help='Kullanılacak model dosyasının yolu')
    args = parser.parse_args()
    play_ai(args.model) 