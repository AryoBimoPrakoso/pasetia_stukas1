import pandas as pd
import numpy as np

df = pd.read_csv("users_256.csv")

n = len(df)

matrix = np.zeros((n,n),dtype=np.int32)

def similarity(a,b):

    score = 0

    # --------------------
    # UMUR
    # --------------------

    selisih = abs(a["Umur"]-b["Umur"])

    if selisih <=5:
        score +=5

    elif selisih<=10:
        score +=3

    elif selisih<=20:
        score +=1


    # --------------------
    # GENDER
    # --------------------

    if a["Gender"]==b["Gender"]:
        score +=1


    # --------------------
    # DAERAH
    # --------------------

    if a["Daerah"]==b["Daerah"]:
        score +=7


    # --------------------
    # KATEGORI
    # --------------------

    if a["Kategori"]==b["Kategori"]:
        score +=6

    return score

for i in range(n):

    user_a = df.iloc[i]

    for j in range(n):

        user_b = df.iloc[j]

        matrix[i,j] = similarity(user_a,user_b)

pd.DataFrame(matrix).to_csv(
    "similarity_matrix_256.csv",
    index=False,
    header=False
)