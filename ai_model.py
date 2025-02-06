import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from direction import Direction
from constants import GRID_SIZE

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

class SnakeAI:
    def __init__(self, state_size=21, hidden_size=256, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.98
        self.epsilon = 1.0  # Başlangıç epsilon değeri
        self.epsilon_min = 0.02  # Minimum epsilon değeri (0.05'ten 0.02'ye)
        self.epsilon_decay = 0.998  # Orta seviye azalma (0.999'dan 0.998'e)
        self.learning_rate = 0.0005
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_model = DQN(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self, game):
        head = game.snake[0]
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # State vektörü (toplam 21 özellik):
        state = [
            # Tehlike düz (1 özellik)
            int((dir_r and self.is_collision(game, point_r)) or 
                (dir_l and self.is_collision(game, point_l)) or 
                (dir_u and self.is_collision(game, point_u)) or 
                (dir_d and self.is_collision(game, point_d))),

            # Tehlike sağ (1 özellik)
            int((dir_u and self.is_collision(game, point_r)) or 
                (dir_d and self.is_collision(game, point_l)) or 
                (dir_l and self.is_collision(game, point_u)) or 
                (dir_r and self.is_collision(game, point_d))),

            # Tehlike sol (1 özellik)
            int((dir_d and self.is_collision(game, point_r)) or 
                (dir_u and self.is_collision(game, point_l)) or 
                (dir_r and self.is_collision(game, point_u)) or 
                (dir_l and self.is_collision(game, point_d))),

            # Hareket yönü (4 özellik)
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),

            # Elma konumu (4 özellik)
            int(game.apple[0] < head[0]),  # elma sol
            int(game.apple[0] > head[0]),  # elma sağ
            int(game.apple[1] < head[1]),  # elma yukarı
            int(game.apple[1] > head[1]),  # elma aşağı

            # Elma mesafesi (2 özellik)
            abs(game.apple[0] - head[0]) / GRID_SIZE,  # x mesafesi
            abs(game.apple[1] - head[1]) / GRID_SIZE,  # y mesafesi

            # Yılanın mevcut yönü (1 özellik)
            int(game.direction.value) / 4.0,  # Normalize edilmiş yön değeri

            # Yılan vücut bilgileri (6 özellik)
            len(game.snake) / GRID_SIZE,  # Normalize edilmiş yılan uzunluğu
            
            # Vücut parçaları var mı? (4 yön)
            int(point_l in game.snake[1:]),  # Sol
            int(point_r in game.snake[1:]),  # Sağ
            int(point_u in game.snake[1:]),  # Yukarı
            int(point_d in game.snake[1:]),  # Aşağı

            # Kuyruk yönü (1 özellik)
            int(game.snake[-1][0] < head[0]) - int(game.snake[-1][0] > head[0]),  # X ekseni kuyruk yönü
            int(game.snake[-1][1] < head[1]) - int(game.snake[-1][1] > head[1])   # Y ekseni kuyruk yönü
        ]

        return np.array(state, dtype=np.float32)

    def is_collision(self, game, point):
        # Duvarlarla çarpışma kontrolü
        if point[0] < 0 or point[0] >= GRID_SIZE or \
           point[1] < 0 or point[1] >= GRID_SIZE:
            return True
        # Yılanın kendisiyle çarpışma kontrolü
        if point in game.snake[1:]:
            return True
        return False

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        # Minibatch'i numpy array'e dönüştür
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # Numpy array'leri tensor'e dönüştür
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Mevcut Q değerleri
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Hedef Q değerleri
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # Kayıp hesaplama ve optimizasyon
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon değerini güncelle
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def check_epsilon_reset(self):
        """Belirli sayıda oyun sonrası epsilon değerini yeniden ayarlar"""
        self.current_game += 1
        if self.current_game % self.games_before_reset == 0:
            self.epsilon = max(self.epsilon_reset_value, self.epsilon)
            print(f"\nEpsilon reset edildi: {self.epsilon:.3f}") 