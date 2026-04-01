import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

veri_yolu = "veri_seti"
model_kayit_yolu = "dinamik_resnet_modeli.pth"
cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transfer Öğrenme için standart ImageNet normalizasyon değerleri
egitim_donusumleri = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Veri seti taranıyor ve dinamik olarak yükleniyor...")
# ImageFolder klasör hiyerarşisini otomatik okur ve etiketler
tam_veri_seti = datasets.ImageFolder(root=veri_yolu, transform=egitim_donusumleri)

# Veri setini Eğitim (%80) ve Doğrulama (%20) olarak dinamik bölüyoruz
egitim_boyutu = int(0.8 * len(tam_veri_seti))
dogrulama_boyutu = len(tam_veri_seti) - egitim_boyutu
egitim_seti, dogrulama_seti = torch.utils.data.random_split(tam_veri_seti, [egitim_boyutu, dogrulama_boyutu])

egitim_yukleyici = DataLoader(egitim_seti, batch_size=32, shuffle=True)
dogrulama_yukleyici = DataLoader(dogrulama_seti, batch_size=32, shuffle=False)
sinif_isimleri = tam_veri_seti.classes

print(f"Toplam Görüntü: {len(tam_veri_seti)} | Eğitim: {egitim_boyutu} | Doğrulama: {dogrulama_boyutu}")
print(f"Tespit Edilen Sınıflar: {sinif_isimleri}")

# ResNet18 Ön Eğitilmiş Modelini Yüklüyoruz
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# ResNet'in son karar katmanını (fully connected layer) bizim 5 sınıfımıza göre değiştiriyoruz
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(sinif_isimleri))

# Sürekli Öğrenme (Continual Learning) Mantığı
if os.path.exists(model_kayit_yolu):
    print(f"\n[BİLGİ] '{model_kayit_yolu}' bulundu. Önceki eğitim hafızası yükleniyor...")
    model.load_state_dict(torch.load(model_kayit_yolu, weights_only=True))
    print("Yeni eklenen hekim verileri modelin üzerine eğitilecek (Fine-tuning).")
else:
    print("\n[BİLGİ] Kayıtlı model bulunamadı. ResNet18 tabanlı ilk eğitim başlatılıyor.")

model = model.to(cihaz)
kriter = nn.CrossEntropyLoss()
# Öğrenme oranını (lr) çok düşük tutuyoruz ki daha önce öğrenilenler unutulmasın
optimizator = optim.Adam(model.parameters(), lr=0.0001)

epoch_sayisi = 10
en_iyi_val_kaybi = float('inf')

print(f"\nAktif Donanım: {cihaz} | Dinamik Eğitim Başlıyor...")

for epoch in range(epoch_sayisi):
    model.train()
    egitim_kaybi = 0.0
    dogru_tahmin = 0
    toplam_veri = 0

    for batch_x, batch_y in egitim_yukleyici:
        batch_x, batch_y = batch_x.to(cihaz), batch_y.to(cihaz)

        optimizator.zero_grad()
        tahminler = model(batch_x)
        kayip = kriter(tahminler, batch_y)
        kayip.backward()
        optimizator.step()

        egitim_kaybi += kayip.item()
        _, o_anki_tahmin = torch.max(tahminler, 1)
        dogru_tahmin += (o_anki_tahmin == batch_y).sum().item()
        toplam_veri += batch_y.size(0)

    epoch_egitim_kaybi = egitim_kaybi / len(egitim_yukleyici)
    epoch_egitim_dogrulugu = (dogru_tahmin / toplam_veri) * 100

    model.eval()
    dogrulama_kaybi = 0.0
    val_dogru_tahmin = 0
    val_toplam_veri = 0

    with torch.no_grad():
        for val_x, val_y in dogrulama_yukleyici:
            val_x, val_y = val_x.to(cihaz), val_y.to(cihaz)
            val_tahminler = model(val_x)
            v_kayip = kriter(val_tahminler, val_y)

            dogrulama_kaybi += v_kayip.item()
            _, val_o_anki_tahmin = torch.max(val_tahminler, 1)
            val_dogru_tahmin += (val_o_anki_tahmin == val_y).sum().item()
            val_toplam_veri += val_y.size(0)

    epoch_val_kaybi = dogrulama_kaybi / len(dogrulama_yukleyici)
    epoch_val_dogrulugu = (val_dogru_tahmin / val_toplam_veri) * 100

    print(
        f"Epoch [{epoch + 1}/{epoch_sayisi}] | Eğt. Kayıp: {epoch_egitim_kaybi:.4f} - Doğr.: %{epoch_egitim_dogrulugu:.2f} | Val. Kayıp: {epoch_val_kaybi:.4f} - Doğr.: %{epoch_val_dogrulugu:.2f}")

    if epoch_val_kaybi < en_iyi_val_kaybi:
        en_iyi_val_kaybi = epoch_val_kaybi
        torch.save(model.state_dict(), model_kayit_yolu)
        print(" --> Ağ güncellendi ve yeni hücre morfolojileri hafızaya kazındı.")

print("\nDinamik eğitim döngüsü tamamlandı. Model klinik tarama için hazır.")