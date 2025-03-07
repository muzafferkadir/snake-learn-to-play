# Snake Game with AI

Bu proje, klasik yılan oyununun Python ile gerçekleştirilmiş ve yapay zeka ile geliştirilmiş bir versiyonudur. Proje iki ana bileşenden oluşur:

1. **Klasik Yılan Oyunu**
   - 32x32 oyun alanı
   - Izgara görünümlü arka plan
   - Yeşil yılan ve kırmızı elma
   - Her 5 elmada bir artan hız
   - Çarpışma kontrolü (duvarlar ve yılanın kendisi)

2. **Yapay Zeka Kontrolü**
   - Deep Q-Learning algoritması
   - Deneyim tekrarı (Experience Replay)
   - Epsilon-greedy keşif stratejisi
   - Eğitim istatistikleri görüntüleme
   - Ayarlanabilir eğitim hızı (1x - 512x)
   - Geliştirilmiş durum algılama (29 özellik)
   - Çıkmaz durumları engelleme
   - 4 katmanlı derin sinir ağı mimarisi

## Yapay Zeka İyileştirmeleri

1. **Genişletilmiş Durum Vektörü (21→29 özellik)**
   - Çapraz yönlerde engel algılama
   - Duvar mesafesi bilgisi
   - Çıkmaz durum tespiti

2. **İyileştirilmiş Ödül Fonksiyonu**
   - Çıkmaz durumlar için özel ceza
   - Elmaya yaklaşmayı daha iyi ödüllendirme
   - Kendi kuyruğuna yaklaşmayı cezalandırma

3. **Güçlendirilmiş Model Mimarisi**
   - 4 katmanlı derin sinir ağı
   - 512 boyutlu gizli katmanlar
   - Daralan mimari yapı

4. **Optimize Edilmiş Hiperparametreler**
   - Daha büyük bellek kapasitesi (200.000)
   - Daha yüksek indirim faktörü (0.995)
   - Daha dengeli öğrenme oranı (0.0002)

## Kurulum

1. Python 3.8 veya daha yüksek bir sürümün yüklü olduğundan emin olun

2. Proje dizininde bir virtual environment oluşturun:
```bash
# macOS/Linux
python3 -m venv venv

# Windows
python -m venv venv
```

3. Virtual environment'ı aktif edin:
```bash
# macOS/Linux
source venv/bin/activate

# Windows
.\venv\Scripts\activate
```

4. Gerekli bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

### Manuel Oyun
Klasik yılan oyununu oynamak için:
```bash
python game.py
```

**Kontroller:**
- Yön tuşları: Yılanı kontrol etme
- ESC: Oyundan çıkış

### AI Eğitimi
Yapay zekayı eğitmek için:
```bash
python train.py
```

### AI ile Oyunu Oynatma
Eğitim sonrası kayıt edilen modeller ile (models klasörü içinde) oyunu başlatmak için:

```bash
python play_ai.py --model models/model_record_***.pth 
```

**Eğitim Parametreleri:**
- `n_games`: Eğitilecek oyun sayısı (varsayılan: 2000)
- `batch_size`: Her adımda eğitilecek örnek sayısı (varsayılan: 64)
- `target_update_freq`: Hedef ağ güncelleme sıklığı (varsayılan: 50 adım)

**Eğitim Ekranı:**
- Sol tarafta oyun alanı
- Sağ tarafta eğitim istatistikleri:
  - Mevcut skor
  - Rekor skor
  - Ortalama skor
  - Epsilon değeri
  - Adım sayısı
  - FPS

**Hız Kontrolü:**
- Eğitim hızını ayarlamak için sağ alt köşedeki butonları kullanın:
  - 1x: Normal hız
  - 8x: 8 kat hızlı
  - 16x: 16 kat hızlı
  - 128x: 128 kat hızlı
  - 512x: 512 kat hızlı

## Proje Yapısı

```
.
├── game.py           # Oyun motoru ve arayüzü
├── ai_model.py       # Yapay zeka modeli (DQN)
├── train.py          # Eğitim döngüsü
├── play_ai.py        # Eğitilmiş model ile oynatma
├── direction.py      # Yön enumları
├── constants.py      # Sabitler
├── requirements.txt  # Bağımlılıklar
├── models/           # Eğitilmiş modeller
└── README.md         # Dokümantasyon
```

## Geliştirme

Yeni paketler eklemek için:
```bash
pip install paket_adi
pip freeze > requirements.txt
```

## Notlar

- `.gitignore` dosyası sayesinde `venv` klasörü ve `__pycache__` git'e dahil edilmez
- Her yeni geliştirme ortamında kurulum adımlarını tekrarlayın
- Eğitim sırasında model ağırlıkları otomatik olarak kaydedilir
- Daha iyi sonuçlar için eğitimi daha uzun süre çalıştırın (n_games > 2000)