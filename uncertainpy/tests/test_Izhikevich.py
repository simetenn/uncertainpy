from uncertainpy.models import IzhikevichModel

import pylab as plt

model = IzhikevichModel()

t, U = model.run()

plt.plot(t, U)
plt.show()
