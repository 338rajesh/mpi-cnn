import os
import sys
from tqdm import tqdm
import numpy as np

CSD = os.path.dirname(__file__)
BDR = os.path.dirname(CSD)
sys.path.insert(0, BDR)
import mpi_cnn

BATCH_SIZE = 64
PRINT_MODEL_SUMMARY = False
E_MIN, E_MAX = 1.0e09, 500.0e09
NU = NU_MIN = NU_MAX = 0.25  # because we have assumed equal P_ratio


DATA_SETS_DIR = os.path.join(BDR, "0_datasets", "containers", "xy_ext_irregular")
MODELS_DIR = os.path.join(CSD, "best_trained_models")
RESULTS_DIR = os.path.join(CSD, "Test-Results")
os.makedirs(RESULTS_DIR, exist_ok=True)


ext_data_set_ids = ("ds2", "ds3", "ds4")

for a_ds_id in ext_data_set_ids:
    RVE_MIA = mpi_cnn.load_data_set(
        os.path.join(DATA_SETS_DIR, f"ext_{a_ds_id}_rve_testing_images.npy"),
        permute_dims=(0, 3, 1, 2),
        merge_chunks=True,
    )
    RVE_META = mpi_cnn.load_data_set(
        os.path.join(DATA_SETS_DIR, f"ext_{a_ds_id}_rve_testing_labels.npy"),
        merge_chunks=True,
    )
    #
    test_results = {}
    for a_model_name in tqdm(os.listdir(MODELS_DIR)):  # NEX_10000-R_1
        nx_id, rl_id = a_model_name.split("-")
        nex_tr = int(nx_id.split("_")[1])
        real_num = int(rl_id.split("_")[1])
        #
        nex_ts = RVE_META.shape[0]
        anx_mia = RVE_MIA[:, 0:1]
        anx_meta = RVE_META
        true_prop = RVE_META[:, 5:8]
        vf = RVE_META[:, 0:1]
        em = RVE_META[:, 1:2]
        ef = RVE_META[:, 3:4]
        cnn_model = mpi_cnn.load_model(
            os.path.join(MODELS_DIR, a_model_name, "best_model.pth"),
            inc=2,  # number of input channels
            ouu=3,  # number of output units
            els_mod_min=E_MIN,
            els_mod_max=E_MAX,
            p_ratio=NU,
            print_summary=False,
        )
        test_dl = mpi_cnn.make_data_loaders(
            (anx_mia, true_prop, vf, em, ef), BATCH_SIZE, nex_ts, shuffle=False
        )
        predictions = cnn_model.predictions_on(test_dl)
        test_results[a_model_name] = mpi_cnn.torch.concatenate(
            (anx_meta[:, :8], predictions), dim=1
        ).cpu().numpy()
        del cnn_model

    np.savez_compressed(
        os.path.join(RESULTS_DIR, f"test_results-{a_ds_id}.npz"),
        **test_results,
    )

    del RVE_MIA, RVE_META
