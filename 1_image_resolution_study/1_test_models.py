import os
import sys
from tqdm import tqdm
import numpy as np

CSD = os.path.dirname(__file__)
BDR = os.path.dirname(CSD)
sys.path.insert(0, BDR)
import mpi_cnn

my_device = mpi_cnn.my_device
#
PX, PY = 256, 256
BATCH_SIZE = 64
PRINT_MODEL_SUMMARY = False
E_MIN, E_MAX = 1.0e09, 500.0e09
NU = NU_MIN = NU_MAX = 0.25  # because we have assumed equal P_ratio
PIXELS = (512, 256, 128, 64, 32,)
NUM_REALIZATIONS = [0, 1, 2, 3, 4, ]
NEX_TOTAL = (2500, 1500, 500)
bounds_type = None

DATA_SETS_DIR = os.path.join(BDR, "0_datasets", "containers", "xy_image_size_study")
MODELS_DIR = os.path.join(CSD, "trained_models")
RESULTS_DIR = os.path.join(CSD, "Test-Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

test_results = {}
for a_model_name in tqdm(os.listdir(MODELS_DIR)):  # NEX_500-R_0-P_64
    nx_id, rl_id, px_id = a_model_name.split("-")
    nex_tr = int(nx_id.split("_")[1])
    real_num = int(rl_id.split("_")[1])
    pix_size = int(px_id.split("_")[1])
    RVE_MIA = mpi_cnn.load_data_set(
        os.path.join(DATA_SETS_DIR, f"rve_iss_{pix_size}x{pix_size}_testing_images.npy"),
        permute_dims=(0, 3, 1, 2)
    )
    RVE_META = mpi_cnn.load_data_set(os.path.join(DATA_SETS_DIR, f"rve_iss_testing_labels.npy"))
    nex_ts = 1000  # nex_tr // 2
    anx_mia = RVE_MIA[:nex_ts, 0:1]
    anx_meta = RVE_META[:nex_ts]
    cnn_model = mpi_cnn.load_model(
        os.path.join(MODELS_DIR, a_model_name, "best_model.pth"),
        inc=2, ouu=3,
        els_mod_min=E_MIN, els_mod_max=E_MAX,
        p_ratio=NU, print_summary=False,
    )
    test_dl = mpi_cnn.make_data_loaders(
        (anx_mia, anx_meta[:, 5:8], anx_meta[:, 0:1], anx_meta[:, 1:2], anx_meta[:, 3:4]),
        BATCH_SIZE, nex_ts, shuffle=False
    )
    predictions = cnn_model.predictions_on(test_dl)
    test_results[a_model_name] = mpi_cnn.torch.concatenate((anx_meta[:, :8], predictions), dim=1).cpu().numpy()
    del cnn_model

np.savez_compressed(os.path.join(RESULTS_DIR, "test_results.npz"), **test_results)
