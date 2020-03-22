import numpy as np
import pylab as plt

from icecube.mcweights import UprightCylinder,ZenithBias
c = UprightCylinder(1200.0,600.0,0.00,1.00,ZenithBias.ProjectedCylinderArea)
d = UprightCylinder(1200.0,600.0,0.00,1.00,ZenithBias.UniformCosZenith)

z = np.linspace(0,1,1000)

print(c.area(0))
print(c.area(1))

area0=c.area(0)
area1=[d.area(x) for x in z]

print(min(area1),max(area1),area0)

min(area1)/area0,max(area1)/area0

plt.plot(z,[a/area0 for a in area1])
plt.ylabel("Relative Area")
plt.xlabel("cos(Zenith)")
plt.show()
