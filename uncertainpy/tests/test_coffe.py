from CoffeeCup2DModel import CoffeeCup2DModel

import pylab as plt

model = CoffeeCup2DModel()

t, U = model.run()

print U.shape
print U



plt.plot(t, U[:,0])
plt.show()
