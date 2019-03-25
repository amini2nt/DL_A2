import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import make_interp_spline, BSpline


def plotThis(x, label ):
	xnew = np.linspace(np.arange(x.shape[0]).min(),np.arange(x.shape[0]).max(),300)
	spl = make_interp_spline(np.arange(x.shape[0]),x, k=3)
	power_smooth = spl(xnew)
	xx = plt.plot(xnew,power_smooth, label = label)



plotThis(allEpochsEvaluationLoss, label = 'sigmoid')
plt.title("evaluation loss")
plt.xlabel("")
plt.ylabel("")
plt.legend()
plt.show()