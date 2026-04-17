import matplotlib.pyplot as plt

# 데이터
years = list(range(2002, 2022))
korea = [0, 5, 0, 0, 2, 4, 2, 3, 2, 2, 7, 5, 4, 5, 6, 9, 4, 7, 4, 2]
usa = [18, 22, 33, 17, 16, 24, 15, 17, 11, 15, 27, 20, 25, 47, 72, 67, 99, 126, 105, 59]
china = [2, 9, 2, 8, 3, 12, 5, 16, 18, 19, 15, 32, 30, 54, 82, 89, 112, 110, 104, 111]
europe = [11, 10, 2, 2, 4, 8, 3, 4, 4, 5, 6, 5, 3, 11, 19, 21, 15, 19, 29, 2]
japan = [23, 16, 11, 16, 16, 12, 17, 9, 9, 5, 5, 10, 12, 7, 5, 7, 11, 15, 21, 5]
pct = [15, 13, 3, 6, 4, 11, 8, 11, 6, 8, 9, 12, 8, 15, 29, 32, 25, 30, 68, 43]

# 그래프
plt.figure(figsize=(12, 6))
plt.plot(years, korea, marker='o', linewidth=2, label='Korea')
plt.plot(years, usa, marker='o', linewidth=2, label='USA')
plt.plot(years, china, marker='o', linewidth=2, label='China')
plt.plot(years, europe, marker='o', linewidth=2, label='Europe')
plt.plot(years, japan, marker='o', linewidth=2, label='Japan')
plt.plot(years, pct, marker='o', linewidth=2, label='PCT')

# 꾸미기 (프레젠테이션용)
plt.title("Patent Applications in Quantum Sensors by Year", fontsize=18, fontweight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of Applications", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.xticks(years, rotation=45)
plt.tight_layout()
plt.show()
