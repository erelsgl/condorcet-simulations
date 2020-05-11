import matplotlib
# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
matplotlib.use('TkAgg')  # No module named '_tkinter'
# matplotlib.use('agg')  # non-GUI backend
# matplotlib.use('pdf')  # non-GUI backend
# matplotlib.use('svg')    # non-GUI backend
# matplotlib.use('ps')    # non-GUI backend
# matplotlib.use('cairo')    # cairo backend requires that pycairo>=1.11.0 or cairocffiis installed
# matplotlib.use('GTK3Agg')  # cairo backend requires that pycairo>=1.11.0 or cairocffiis installed
# matplotlib.use('Qt5Agg')     # Failed to import any qt binding
# matplotlib.use('Qt4Agg')     # Failed to import any qt binding
# matplotlib.use('nbAgg')     # No module named 'IPython'
# matplotlib.use('pgf')     # non-GUI backend
# matplotlib.use('WX')     # No module named 'wx'
import matplotlib.pyplot as plt
plt.plot(range(5), range(5))
plt.show()

