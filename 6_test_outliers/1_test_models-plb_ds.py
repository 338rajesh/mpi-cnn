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

BOUNDS_TYPES = ("NOB", "HS")

DATA_SETS_DIR = os.path.join(BDR, "0_datasets", "containers", "xy_plb")
RESULTS_DIR = os.path.join(CSD, "Test-Results-PLB")
os.makedirs(RESULTS_DIR, exist_ok=True)

#
RVE_MIA = mpi_cnn.load_data_set(
    os.path.join(DATA_SETS_DIR, f"plb_ds-rve_testing_images.npy"),
    permute_dims=(0, 3, 1, 2),
    merge_chunks=True,
)
RVE_META = mpi_cnn.load_data_set(
    os.path.join(DATA_SETS_DIR, f"plb_ds-rve_testing_labels.npy"),
    merge_chunks=True,
)
#
test_results = {}
for a_bound_type in BOUNDS_TYPES:
    MODELS_DIR = os.path.join(CSD, "trained_models", a_bound_type)
    for a_model_name in tqdm(os.listdir(MODELS_DIR)):  # NEX_500-R_1
        model_identifiers = a_model_name.split("-")
        nx_id, rl_id = model_identifiers[0], model_identifiers[1]
        #
        nex_tr = int(nx_id.split("_")[1])
        real_num = int(rl_id.split("_")[1])
        #
        nex_ts = 10_000
        anx_mia = RVE_MIA[:nex_ts, 0:1]
        anx_meta = RVE_META[:nex_ts]
        true_prop = anx_meta[:, 5:8]
        vf = anx_meta[:, 0:1]
        em = anx_meta[:, 1:2]
        ef = anx_meta[:, 3:4]
        cnn_model = mpi_cnn.load_model(
            os.path.join(MODELS_DIR, a_model_name, "best_model.pth"),
            inc=2,  # number of input channels
            ouu=3,  # number of output units
            els_mod_min=E_MIN,
            els_mod_max=E_MAX,
            p_ratio=NU,
            print_summary=False,
            bounds_type=a_bound_type,
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
        os.path.join(RESULTS_DIR, f"test_results-{a_bound_type}.npz"),
        **test_results,
    )
