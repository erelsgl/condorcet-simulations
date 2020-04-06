import numpy as np
from scipy.stats import truncnorm

def bounds_for_expertise(mean:float, std:float)->(float,float):
       scale = std
       loc = mean
       lower = 0.5
       upper = 1
       # a*scale + loc = lower
       # b*scale + loc = upper
       a = (lower-loc)/scale
       b = (upper-loc)/scale
       return (a,b)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

mean, std = 0.7, 0.3
a, b = bounds_for_expertise(mean, std)
print(a,b)
scale, loc  = std, mean

x = np.linspace(0, 2, 100)
ax.plot(x, truncnorm.pdf(x, a, b, loc=loc, scale=scale),
       'r-', lw=5, alpha=0.6, label='truncnorm pdf')

# rv = truncnorm(a, b)
# ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
#
r = truncnorm.rvs(a, b, loc=loc, scale=scale, size=100)
print(r)
ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()

