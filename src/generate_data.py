import random
import pandas as pd

JUMLAH_USER = 256

gender = ["L", "P"]

daerah = [
    "Jakarta","Bandung","Surabaya","Medan",
    "Makassar","Bogor","Depok","Bekasi",
    "Semarang","Yogyakarta","Tangerang",
    "Palembang","Balikpapan","Pontianak",
    "Denpasar","Padang","Batam","Manado"
]

kategori = [
    "Film",
    "Musik",
    "Game",
    "Olahraga",
    "Kuliner",
    "Travel",
    "Teknologi",
    "Fashion",
    "Edukasi"
]

rows = []

for i in range(JUMLAH_USER):

    rows.append({

        "UserID":f"U{i+1:04d}",
        "Umur":random.randint(17,80),
        "Gender":random.choice(gender),
        "Daerah":random.choice(daerah),
        "Kategori":random.choice(kategori)

    })

df = pd.DataFrame(rows)

df.to_csv(
    "users_256.csv",
    index=False,
    encoding="utf-8-sig"
)

print(df.head())