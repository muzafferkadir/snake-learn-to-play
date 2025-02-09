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

**Eğitim Parametreleri:**
- `n_games`: Eğitilecek oyun sayısı (varsayılan: 1000)
- `batch_size`: Her adımda eğitilecek örnek sayısı (varsayılan: 32)

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
├── direction.py      # Yön enumları
├── constants.py      # Sabitler
├── requirements.txt  # Bağımlılıklar
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
- Daha iyi sonuçlar için eğitimi daha uzun süre çalıştırın (n_games > 1000)