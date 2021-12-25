import matplotlib.pyplot as plt
import numpy as np
import math

def lagrange_interpolation_polynomial(dot_x, array_x, array_y):
    y = 0
    for i in range(len(array_x)):
        l = 1
        for j in range(len(array_x)):
            if i != j:
                l *= (dot_x-array_x[j])/(array_x[i]-array_x[j])
        y += array_y[i] * l;
    return y

def chebishef(min, max, quantity):
    uniform_arr = []
    i = 1
    while i <= quantity:
        uniform_arr.append(1/2*(min+max) + 1/2*(max-min)*math.cos((2*i-1)*math.pi/(2*quantity)))
        i+=1
    return uniform_arr



# x = np.linspace(0, 10, 2) # 1 задание
# x = chebishef(0., 10., 10) # чебышев
x2 = np.linspace(0, 10, 50)

print(x)
print("")
y = [math.cos(i/3)*math.cos(i/3)/(1+i*i) for i in x]
y2 = [math.cos(i/3)*math.cos(i/3)/(1+i*i) for i in x2]

x1 = np.linspace(0, 10, 50)
print(x1)
print(len(x1))

y1 = []
for i in x1:
    y1.append(lagrange_interpolation_polynomial(i, x, y))
print (y1)

# Построение графика
plt.title("cos(i/3)*cos(i*3)/(1+i*i)    19 задание") # заголовок
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.plot(x, y, 'ro')
plt.plot(x2, y2, x1, y1)
plt.show()
