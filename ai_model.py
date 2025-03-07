# Gerekli kütüphanelerin import edilmesi
import torch  # PyTorch - derin öğrenme framework'ü
import torch.nn as nn  # Sinir ağı modülleri
import torch.optim as optim  # Optimizasyon algoritmaları
import numpy as np  # Numerik işlemler için
import random  # Rastgele sayı üretimi için
from collections import deque  # Deneyim hafızası için çift yönlü kuyruk
from direction import Direction  # Yön enumları
from constants import GRID_SIZE  # Oyun sabitleri

class DQN(nn.Module):
    """Deep Q-Network (DQN) sınıfı
    
    Bu sınıf, derin Q-öğrenme için kullanılan sinir ağı modelini tanımlar.
    4 katmanlı bir yapıya sahiptir: giriş katmanı -> gizli katmanlar -> çıkış katmanı
    """
    def __init__(self, input_size, hidden_size, output_size):
        """DQN modelinin yapılandırılması
        
        Args:
            input_size (int): Giriş katmanı boyutu (durum vektörü boyutu)
            hidden_size (int): Gizli katman boyutu
            output_size (int): Çıkış katmanı boyutu (aksiyon sayısı)
        """
        super(DQN, self).__init__()
        # Geliştirilmiş sinir ağı mimarisi
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Giriş -> 1. Gizli katman
            nn.ReLU(),  # Aktivasyon fonksiyonu
            nn.Linear(hidden_size, hidden_size),  # 1. Gizli -> 2. Gizli katman
            nn.ReLU(),  # Aktivasyon fonksiyonu
            nn.Linear(hidden_size, hidden_size // 2),  # 2. Gizli -> 3. Gizli katman (daha küçük)
            nn.ReLU(),  # Aktivasyon fonksiyonu
            nn.Linear(hidden_size // 2, output_size)  # 3. Gizli -> Çıkış katmanı
        )

    def forward(self, x):
        """İleri yayılım işlemi
        
        Args:
            x (torch.Tensor): Giriş tensörü (durum)
            
        Returns:
            torch.Tensor: Q-değerleri tensörü
        """
        return self.network(x)

class SnakeAI:
    """Yılan AI sınıfı
    
    Bu sınıf, yılanın davranışlarını kontrol eden yapay zeka ajanını temsil eder.
    DQN algoritması kullanarak yılanın optimal hareketleri öğrenmesini sağlar.
    """
    def __init__(self, state_size=29, hidden_size=512, action_size=3):
        """SnakeAI sınıfının başlatılması
        
        Args:
            state_size (int): Durum vektörünün boyutu (default: 29)
            hidden_size (int): Gizli katman boyutu (default: 512)
            action_size (int): Aksiyon uzayı boyutu (default: 3)
        """
        # Temel parametreler
        self.state_size = state_size  # Durum vektörü boyutu
        self.action_size = action_size  # Aksiyon sayısı
        self.memory = deque(maxlen=200000)  # Deneyim hafızası (son 200,000 deneyim)
        
        # Öğrenme parametreleri
        self.gamma = 0.995  # İndirim faktörü (gelecek ödüllerin ağırlığı)
        self.epsilon = 1.0  # Başlangıç keşif oranı
        self.epsilon_min = 0.01  # Minimum keşif oranı
        self.epsilon_decay = 0.9995  # Keşif oranı azalma katsayısı
        self.learning_rate = 0.0002  # Öğrenme oranı
        
        # Cihaz seçimi (GPU varsa GPU, yoksa CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ana model ve hedef model oluşturma
        self.model = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_model = DQN(state_size, hidden_size, action_size).to(self.device)
        
        # Optimizer tanımlama (Adam optimizer)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Hedef modeli ana model ile senkronize et
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self, game):
        """Oyunun mevcut durumunu bir durum vektörüne dönüştürür
        
        Bu fonksiyon, oyunun mevcut durumunu AI'nın anlayabileceği bir formata çevirir.
        Toplam 29 özellikten oluşan bir durum vektörü oluşturur.
        
        Args:
            game: Oyun nesnesi
            
        Returns:
            numpy.array: 29 elemanlı durum vektörü
        """
        # Yılanın başı ve etrafındaki noktalar
        head = game.snake[0]  # Yılanın başı
        point_l = (head[0] - 1, head[1])  # Sol nokta
        point_r = (head[0] + 1, head[1])  # Sağ nokta
        point_u = (head[0], head[1] - 1)  # Üst nokta
        point_d = (head[0], head[1] + 1)  # Alt nokta
        
        # Çapraz noktalar (daha geniş algı alanı için)
        point_ul = (head[0] - 1, head[1] - 1)  # Sol üst çapraz
        point_ur = (head[0] + 1, head[1] - 1)  # Sağ üst çapraz
        point_dl = (head[0] - 1, head[1] + 1)  # Sol alt çapraz
        point_dr = (head[0] + 1, head[1] + 1)  # Sağ alt çapraz

        # Yılanın mevcut yönü
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Durum vektörü (toplam 29 özellik):
        state = [
            # 1-3: Tehlike algılama (3 özellik)
            # Önde tehlike var mı?
            int((dir_r and self.is_collision(game, point_r)) or 
                (dir_l and self.is_collision(game, point_l)) or 
                (dir_u and self.is_collision(game, point_u)) or 
                (dir_d and self.is_collision(game, point_d))),

            # Sağda tehlike var mı?
            int((dir_u and self.is_collision(game, point_r)) or 
                (dir_d and self.is_collision(game, point_l)) or 
                (dir_l and self.is_collision(game, point_u)) or 
                (dir_r and self.is_collision(game, point_d))),

            # Solda tehlike var mı?
            int((dir_d and self.is_collision(game, point_r)) or 
                (dir_u and self.is_collision(game, point_l)) or 
                (dir_r and self.is_collision(game, point_u)) or 
                (dir_l and self.is_collision(game, point_d))),

            # 4-7: Hareket yönü (4 özellik)
            int(dir_l),  # Sola mı gidiyor?
            int(dir_r),  # Sağa mı gidiyor?
            int(dir_u),  # Yukarı mı gidiyor?
            int(dir_d),  # Aşağı mı gidiyor?

            # 8-11: Elma konumu (4 özellik)
            int(game.apple[0] < head[0]),  # Elma solda mı?
            int(game.apple[0] > head[0]),  # Elma sağda mı?
            int(game.apple[1] < head[1]),  # Elma yukarıda mı?
            int(game.apple[1] > head[1]),  # Elma aşağıda mı?

            # 12-13: Elma mesafesi (2 özellik)
            abs(game.apple[0] - head[0]) / GRID_SIZE,  # X ekseni mesafesi (normalize edilmiş)
            abs(game.apple[1] - head[1]) / GRID_SIZE,  # Y ekseni mesafesi (normalize edilmiş)

            # 14-17: Duvar yakınlık algısı (4 özellik) - Yeni!
            head[0] / GRID_SIZE,  # Normalize edilmiş X pozisyonu (sol duvar mesafesi)
            (GRID_SIZE - 1 - head[0]) / GRID_SIZE,  # Normalize edilmiş sağ duvar mesafesi
            head[1] / GRID_SIZE,  # Normalize edilmiş Y pozisyonu (üst duvar mesafesi)
            (GRID_SIZE - 1 - head[1]) / GRID_SIZE,  # Normalize edilmiş alt duvar mesafesi

            # 18: Yılanın mevcut yönü (1 özellik)
            int(game.direction.value) / 4.0,  # Normalize edilmiş yön değeri

            # 19: Yılan uzunluğu (1 özellik)
            len(game.snake) / GRID_SIZE,  # Normalize edilmiş yılan uzunluğu
            
            # 20-23: Yakın vücut parçaları (4 özellik)
            int(point_l in game.snake[1:]),  # Solda vücut var mı?
            int(point_r in game.snake[1:]),  # Sağda vücut var mı?
            int(point_u in game.snake[1:]),  # Yukarıda vücut var mı?
            int(point_d in game.snake[1:]),  # Aşağıda vücut var mı?
            
            # 24-27: Çapraz vücut parçaları (4 özellik) - Yeni!
            int(point_ul in game.snake[1:]),  # Sol üst çaprazda vücut var mı?
            int(point_ur in game.snake[1:]),  # Sağ üst çaprazda vücut var mı?
            int(point_dl in game.snake[1:]),  # Sol alt çaprazda vücut var mı?
            int(point_dr in game.snake[1:]),  # Sağ alt çaprazda vücut var mı?

            # 28-29: Kuyruk yönü (2 özellik)
            int(game.snake[-1][0] < head[0]) - int(game.snake[-1][0] > head[0]),  # X ekseni kuyruk yönü
            int(game.snake[-1][1] < head[1]) - int(game.snake[-1][1] > head[1])   # Y ekseni kuyruk yönü
        ]

        return np.array(state, dtype=np.float32)

    def is_collision(self, game, point):
        """Verilen noktada çarpışma olup olmadığını kontrol eder
        
        Args:
            game: Oyun nesnesi
            point (tuple): Kontrol edilecek nokta (x, y)
            
        Returns:
            bool: Çarpışma varsa True, yoksa False
        """
        # Duvarlarla çarpışma kontrolü
        if point[0] < 0 or point[0] >= GRID_SIZE or \
           point[1] < 0 or point[1] >= GRID_SIZE:
            return True
        # Yılanın kendisiyle çarpışma kontrolü (baş hariç tüm vücut)
        if point in game.snake[1:]:
            return True
        return False

    def remember(self, state, action, reward, next_state, done):
        """Deneyimi hafızaya kaydeder (Experience Replay)
        
        Args:
            state: Mevcut durum
            action: Seçilen aksiyon
            reward: Alınan ödül
            next_state: Sonraki durum
            done: Oyun bitti mi?
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Verilen duruma göre bir aksiyon seçer
        
        Epsilon-greedy stratejisi kullanır:
        - epsilon olasılıkla rastgele aksiyon (keşif)
        - 1-epsilon olasılıkla en iyi aksiyon (kullanım)
        
        Args:
            state: Mevcut durum
            
        Returns:
            int: Seçilen aksiyon indeksi
        """
        # Epsilon olasılığıyla rastgele aksiyon seç (keşif)
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # En yüksek Q-değerine sahip aksiyonu seç (kullanım)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():  # Gradient hesaplama
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        """Deneyim tekrarı ile öğrenme gerçekleştirir
        
        Args:
            batch_size (int): Mini-batch boyutu
            
        Returns:
            float: Kayıp değeri veya None (yeterli deneyim yoksa)
        """
        # Yeterli deneyim yoksa öğrenme yapma
        if len(self.memory) < batch_size:
            return None

        # Rastgele deneyim örnekleri seç ve numpy dizilerine dönüştür
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # Numpy dizilerini PyTorch tensörlerine çevir
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Bellman denklemi ile Q-değerlerini güncelle
        # 1. Mevcut durumlar için Q-değerleri
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # 2. Sonraki durumlar için maksimum Q-değerleri (hedef ağ kullanarak)
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        # 3. Hedef Q-değerleri = ödül + (1-done) * gamma * max_next_q
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # Kayıp hesaplama (MSE) ve geri yayılım
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon değerini azalt (keşif oranını düşür)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def load(self, name):
        """Eğitilmiş modeli dosyadan yükler
        
        Args:
            name (str): Model dosyasının yolu
        """
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        """Mevcut modeli dosyaya kaydeder
        
        Args:
            name (str): Kaydedilecek dosyanın yolu
        """
        torch.save(self.model.state_dict(), name)

    def check_epsilon_reset(self):
        """Belirli sayıda oyun sonrası epsilon değerini yeniden ayarlar
        
        Bu fonksiyon, modelin belirli aralıklarla yeniden keşif yapmasını sağlar.
        Böylece yerel optimumlara takılması engellenir.
        """
        self.current_game += 1
        if self.current_game % self.games_before_reset == 0:
            self.epsilon = max(self.epsilon_reset_value, self.epsilon)
            print(f"\nEpsilon reset edildi: {self.epsilon:.3f}") 