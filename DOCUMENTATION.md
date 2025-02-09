# Snake Oyunu ve Derin Pekiştirmeli Öğrenme Projesi

## İçindekiler
1. [Giriş](#giriş)
2. [Makine Öğrenimi ve Derin Öğrenme](#makine-öğrenimi-ve-derin-öğrenme)
3. [Pekiştirmeli Öğrenme](#pekiştirmeli-öğrenme)
4. [Proje Yapısı](#proje-yapısı)
5. [Teknik Detaylar](#teknik-detaylar)
6. [Eğitim Süreci](#eğitim-süreci)
7. [Sonuçlar ve Analiz](#sonuçlar-ve-analiz)
8. [Kurulum ve Kullanım](#kurulum-ve-kullanım)
9. [Gelecek Geliştirmeler](#gelecek-geliştirmeler)
10. [Kaynaklar](#kaynaklar)

## Giriş

Bu proje, klasik Snake oyununun yapay zeka ile oynamayı öğrenmesini sağlayan bir derin pekiştirmeli öğrenme uygulamasıdır. Proje, hem eğlenceli bir oyun deneyimi sunmakta hem de modern yapay zeka tekniklerinin pratik bir uygulamasını göstermektedir.

## Makine Öğrenimi ve Derin Öğrenme

### Makine Öğrenimi Nedir?

Makine öğrenimi, bilgisayarların açık programlama olmadan verilerden öğrenmesini sağlayan yapay zeka alt dalıdır. Temel olarak üç ana kategoriye ayrılır:

1. **Denetimli Öğrenme (Supervised Learning)**
   - Etiketli veri setleri kullanır
   - Örnek: Görüntü sınıflandırma, spam tespiti

2. **Denetimsiz Öğrenme (Unsupervised Learning)**
   - Etiketsiz veri setleri kullanır
   - Örnek: Müşteri segmentasyonu, anomali tespiti

3. **Pekiştirmeli Öğrenme (Reinforcement Learning)**
   - Ödül-ceza mekanizması kullanır
   - Örnek: Oyun oynama, robot kontrolü

### Derin Öğrenme

Derin öğrenme, makine öğreniminin bir alt kümesidir ve çok katmanlı yapay sinir ağları kullanır. Bu projede kullanılan DQN (Deep Q-Network) modeli, derin öğrenme tekniklerini kullanmaktadır.

## Pekiştirmeli Öğrenme

### Temel Kavramlar

1. **Ajan (Agent)**
   - Öğrenen ve karar veren birim
   - Bu projede: Snake'i kontrol eden AI

2. **Çevre (Environment)**
   - Ajanın etkileşimde bulunduğu ortam
   - Bu projede: Oyun tahtası

3. **Durum (State)**
   - Çevrenin anlık durumu
   - Bu projede: Yılanın pozisyonu, elmanın konumu, engeller

4. **Eylem (Action)**
   - Ajanın yapabileceği hareketler
   - Bu projede: Sola dön, sağa dön, düz git

5. **Ödül (Reward)**
   - Ajanın eylemlerine verilen geri bildirim
   - Bu projede: Elma yeme (+10), çarpışma (-10), boş hareket (-0.1)

### DQN (Deep Q-Network)

DQN, Q-öğrenme algoritmasını derin sinir ağları ile birleştiren bir yaklaşımdır. Bu projede kullanılan DQN modeli şu özelliklere sahiptir:

- 3 katmanlı sinir ağı (21 giriş, 256 gizli, 3 çıkış)
- Experience Replay belleği (100,000 deneyim)
- Epsilon-greedy keşif stratejisi
- Target Network kullanımı

## Proje Yapısı

### Dosyalar ve Görevleri

1. **game.py**
   - Oyun mantığı ve görsel arayüz
   - Pygame kütüphanesi ile geliştirilmiş
   - Training ve normal mod desteği

2. **ai_model.py**
   - DQN modelinin implementasyonu
   - PyTorch ile geliştirilmiş
   - Deneyim toplama ve öğrenme mekanizmaları

3. **train.py**
   - Eğitim döngüsü ve loglama
   - Model kaydetme ve yükleme
   - Performans metrikleri

4. **constants.py**
   - Oyun sabitleri
   - Görsel öğeler için renk kodları
   - Grid boyutları ve hız ayarları

### Veri Yapıları

1. **State Vektörü (21 özellik)**
   - Tehlike algılama (3 özellik)
   - Hareket yönü (4 özellik)
   - Elma konumu (4 özellik)
   - Elma mesafesi (2 özellik)
   - Yılan bilgileri (8 özellik)

2. **Action Space (3 eylem)**
   - 0: Düz git
   - 1: Sağa dön
   - 2: Sola dön

## Teknik Detaylar

### Ödül Sistemi

```python
# Ödül hesaplama mantığı
if game_over:
    reward = -10  # Çarpışma cezası
elif new_score > old_score:
    reward = 10   # Elma yeme ödülü
else:
    # Elmaya yaklaşma/uzaklaşma kontrolü
    if new_distance < old_distance:
        reward = 0.1  # Elmaya yaklaşma ödülü
    else:
        reward = -0.1  # Elmadan uzaklaşma cezası
```

### Epsilon-Greedy Stratejisi

```python
self.epsilon = 1.0        # Başlangıç değeri
self.epsilon_min = 0.02   # Minimum değer
self.epsilon_decay = 0.998  # Azalma oranı
```

## Eğitim Süreci

### Eğitim Parametreleri

- Batch Size: 32
- Öğrenme Oranı: 0.0005
- Gamma (İndirim Faktörü): 0.98
- Hedef Ağ Güncelleme: Her 100 adımda bir

### Eğitim Döngüsü

1. **Başlangıç**
   - Model ve oyun ortamı oluşturulur
   - Epsilon maksimum değere ayarlanır

2. **Her Adımda**
   - Durum gözlemlenir
   - Epsilon-greedy ile eylem seçilir
   - Eylem uygulanır ve ödül alınır
   - Deneyim belleğe kaydedilir
   - Mini-batch ile öğrenme yapılır

3. **Her Oyun Sonunda**
   - İstatistikler güncellenir
   - Model kaydedilir (gerekirse)
   - Epsilon değeri azaltılır

## Sonuçlar ve Analiz

### Performans Metrikleri

- Ortalama skor
- Maksimum skor
- Ölüm nedenleri dağılımı
- Epsilon değişimi
- Toplam elma sayısı

### Gözlemler

1. **Öğrenme Eğrileri**
   - İlk 100 oyunda hızlı öğrenme
   - 200-500 arası stabilizasyon
   - 500+ oyunda ince ayar

2. **Davranış Analizi**
   - Duvarlardan kaçınma
   - Kuyruğu yönetme
   - Elma toplama stratejileri

## Kurulum ve Kullanım

### Gereksinimler

```
python 3.8+
pygame
pytorch
numpy
```

### Kurulum Adımları

1. Repo'yu klonlayın
2. Virtual environment oluşturun
3. Gereksinimleri yükleyin
4. Oyunu başlatın

### Kullanım

- Normal mod: `python game.py`
- Eğitim modu: `python train.py`
- AI ile oyna: `python play_ai.py`

## Gelecek Geliştirmeler

1. **Model İyileştirmeleri**
   - Daha karmaşık ağ mimarisi
   - Prioritized Experience Replay
   - Dueling DQN implementasyonu

2. **Oyun Özellikleri**
   - Farklı zorluk seviyeleri
   - Engeller ve özel güçler
   - Çoklu oyuncu modu

3. **Kullanıcı Arayüzü**
   - Gerçek zamanlı grafik gösterimi
   - Daha detaylı istatistikler
   - Eğitim parametrelerini ayarlama arayüzü

## Kaynaklar

1. Deep Q-Learning Paper (Mnih et al., 2015)
2. PyTorch Documentation
3. Pygame Documentation
4. Reinforcement Learning: An Introduction (Sutton & Barto)
