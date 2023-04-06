import os
import sys

CSD = os.path.dirname(__file__)
BDR = os.path.dirname(CSD)
sys.path.insert(0, BDR)
import mpi_cnn

# ============================================================ #
# ============================================================ #

MY_DEVICE = mpi_cnn.my_device
BATCH_SIZE = 64
NUM_EPOCHS = 200
PRINT_MODEL_SUMMARY = False
DATA_SETS_DIR = os.path.join(BDR, "0_datasets", "containers", "xy")
E_MIN, E_MAX = 1.0e09, 500.0e09
NU = NU_MIN = NU_MAX = 0.25  # because we have assumed equal P_ratio
NUM_REALIZATIONS = [3, 4, 5, 6, 7, 8, 9]  # list(range(1, 10,))  # 5, 6, 7, 8, 9,
BOUND_IDS = ("HS",)  # "VRH", "HS"
#
RVE_MIA = mpi_cnn.load_data_set(
    os.path.join(DATA_SETS_DIR, f"rve_training_images.npy"),
    permute_dims=(0, 3, 1, 2)
)
RVE_META = mpi_cnn.load_data_set(os.path.join(DATA_SETS_DIR, f"rve_training_labels.npy"))
#
nex_total = 10_000
for a_real in NUM_REALIZATIONS:
    for a_bound_type in BOUND_IDS:
        #
        rve_mia = RVE_MIA[:nex_total, 0:1]
        rve_meta = RVE_META[:nex_total]
        true_prop = rve_meta[:, 5:8]
        vf = rve_meta[:, 0:1]
        em = rve_meta[:, 1:2]
        ef = rve_meta[:, 3:4]
        #
        nex_train = int(nex_total * 0.80)
        nex_val = nex_total - nex_train
        #
        (Training_Data_Loader, Validation_Data_Loader) = mpi_cnn.make_data_loaders(
            (rve_mia, true_prop, vf, em, ef,),
            BATCH_SIZE, (nex_train, nex_val), shuffle=True
        )
        #
        model_id = f"NEX_{nex_total}-R_{a_real}-B_{a_bound_type}"
        print(f"\n{'_' * 50}\nWorking on {model_id}\n{'_' * 50}")
        model_dir = os.path.join(CSD, "trained_models", model_id)
        os.makedirs(model_dir, exist_ok=True)
        #
        # Building and training the network
        MPI_CNN = mpi_cnn.MpiCNN(
            in_channels=2,
            out_units=3,
            working_dir=model_dir,
            e_min=E_MIN,
            e_max=E_MAX,
            nu=NU,
            bounds_type=a_bound_type
        ).to(MY_DEVICE)
        MPI_CNN.init_weights(method="xavier_uniform")
        if PRINT_MODEL_SUMMARY:
            mpi_cnn.summary(MPI_CNN)
        MPI_CNN.trainer(
            train_dl=Training_Data_Loader,
            optimizer=mpi_cnn.torch.optim.Adam(MPI_CNN.parameters(), lr=5e-04),
            num_epochs=NUM_EPOCHS,
            val_dl=Validation_Data_Loader,
            test_dl=None,
        )
        del MPI_CNN  # deleting the model from memory.
