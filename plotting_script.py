import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import make_interp_spline, BSpline


def fix_times(times_list):
	final_times_list = []
	for i in range(len(times_list)):
		final_times_list[i] = times_list[i]
		for j in range(i):
			final_times_list[i] += times_list[j]
	return final_times_list

lc_path = '/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0/learning_curves.npy'
model = np.load(lc_path)[()]

##    'train_ppls', 'val_losses', 'times', 'train_losses', 'val_ppls'




plt.plot(model['train_ppls'], )
plt.title("RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35")
plt.xlabel("epochs")
plt.ylabel("train_ppls")
#plt.legend(['train_ppl'])
plt.savefig('train_ppls.png')