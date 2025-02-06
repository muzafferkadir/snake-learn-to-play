# Snake Game with AI

Bu proje, klasik yılan oyununun Python ile gerçekleştirilmiş bir versiyonudur. Proje iki aşamadan oluşmaktadır:
1. Klasik yılan oyunu
2. Makine öğrenmesi ile kendini geliştiren AI kontrolü (gelecek aşama)

## Özellikler
- 32x32 oyun alanı
- Izgara görünümlü arka plan
- Yeşil yılan
- Kırmızı elma
- Her 5 elmada bir artan hız
- Çarpışma kontrolü (duvarlar ve yılanın kendisi)

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

5. Oyunu başlatın:
```bash
python game.py
```

6. Oyunu kapattıktan sonra virtual environment'dan çıkmak için:
```bash
deactivate
```

## Kontroller
- Yön tuşları ile yılanı kontrol edebilirsiniz
- ESC tuşu ile oyundan çıkabilirsiniz

## Geliştirme

Virtual environment'ı aktif ettikten sonra yeni paketler eklemek için:
```bash
pip install paket_adi
pip freeze > requirements.txt
```

## Not
- `.gitignore` dosyası sayesinde `venv` klasörü git versiyon kontrolüne dahil edilmeyecektir.
- Her yeni geliştirme ortamında yukarıdaki kurulum adımlarını tekrarlamanız gerekecektir. 