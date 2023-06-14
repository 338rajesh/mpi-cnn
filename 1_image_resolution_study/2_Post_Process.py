import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

CSD = os.path.dirname(__file__)
test_res = np.load(os.path.join(CSD, "Test-Results", "test_results.npz"))


def mape(t, p): return np.mean(np.abs(1 - (p / t)) * 100.0)


markers = {500: 'o', 1500: 'X', 2500: 's'}
colors = {500: 'r', 1500: 'b', 2500: 'k'}
nex_wise_mape = {}
fig, axs = plt.subplots(1, 1, figsize=(8, 8))
for (model_id, model_predictions) in (test_res.items()):
    nx_id, rl_id, px_id = model_id.split("-")
    nex_tr = int(nx_id.split("_")[1])
    real_num = int(rl_id.split("_")[1])
    pix_size = int(px_id.split("_")[1])
    if nex_tr in (500, 1500, 2500):
        #
        true_values = model_predictions[:, 5:8]
        predictions = model_predictions[:, 8:11]
        a_mape = mape(true_values, predictions)
        axs.scatter(pix_size, a_mape, s=50.0, marker=markers[nex_tr], c="None", edgecolors=colors[nex_tr])
        #
        if nex_tr in nex_wise_mape.keys():
            nex_wise_mape[nex_tr].append([pix_size, a_mape])
        else:
            nex_wise_mape[nex_tr] = [[pix_size, a_mape], ]
        #
axs.set_xticks([32, 64, 128, 256, 512])
nex_wise_mape = {k: np.array(v, dtype=np.float_) for (k, v) in nex_wise_mape.items()}
#
for (a_nex_tr, mape_values) in nex_wise_mape.items():

    a_nx_tr_mape = []
    for a_pix in np.unique(mape_values[:, 0]):
        a_pix_mape = mape_values[mape_values[:, 0] == a_pix][:, 1]
        apm_mean, apm_std = a_pix_mape.mean(), a_pix_mape.std()
        a_nx_tr_mape.append([a_pix, apm_mean, apm_std])
    a_nx_tr_mape = np.array(a_nx_tr_mape)
    axs.plot(a_nx_tr_mape[:, 0], a_nx_tr_mape[:, 1], color=colors[a_nex_tr], label=f"{a_nex_tr}")

axs.legend(loc='best')
fig.savefig(os.path.join(CSD, "iss_mape.png"))
