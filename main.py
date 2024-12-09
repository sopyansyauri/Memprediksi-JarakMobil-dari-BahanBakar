import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

df = pd.read_csv("bensin.csv")
liter = df[["Liter"]]
kilometer = df[["Kilometer"]]
X_train, X_test, y_train, y_test = ms.train_test_split(liter, kilometer, test_size=0.2, random_state=0)
# print(X_test, X_test)
plt.scatter(X_train, y_train, edgecolors="r")
plt.xlabel("Liter")
plt.ylabel("Kilometer")
plt.title("Konsumsi bahan bakar")
x1 = np.linspace(0, 45)
y1 = 10.64 + 6.45 * x1
plt.plot(x1, y1)

# membuat model
model1 = lm.LinearRegression()
model1.fit(X_train, y_train)
# print(model1.coef_)
# print(model1.intercept_)
r = model1.score(X_test, y_test)
jarak = model1.predict(X_test)
# print(np.round(jarak))
# plt.show()

print("=====================================================")
print("Memprediksi Jarak dari Mobil kalau di isi Bahan Bakar")
print("=====================================================")
Liter = []

loop = True

while (loop):
    masukan = int(input("Berapa Liter: "))
    Liter.append(masukan)
    data = input("apakah mau ditambah yes/no: ")
    if (data == "yes"):
        loop = True
    elif (data == "no"):
        loop = False
    else:
        break

Liter = pd.DataFrame(Liter, columns=["Liter"])
# print(Liter)
prediksi = np.int_(model1.predict(Liter))
Prediksi = pd.DataFrame(prediksi, columns=["Kilometer"])
# print(Prediksi)

print("\n")
print("==============")
print("Hasil Prediksi")
print("==============")
hasil = pd.DataFrame.join(Liter, Prediksi)
print(hasil)
