import os
import sys

CSD = os.path.dirname(__file__)
BDR = os.path.dirname(CSD)
sys.path.insert(0, BDR)
import mpi_cnn

my_device = mpi_cnn.my_device
#
PX, PY = 256, 256
BATCH_SIZE = 64
NUM_EPOCHS = 200
PRINT_MODEL_SUMMARY = False
DATA_SETS_DIR = os.path.join(BDR, "0_datasets", "containers", "xy_image_size_study")
E_MIN, E_MAX = 1.0e09, 500.0e09
NU = NU_MIN = NU_MAX = 0.25  # because we have assumed equal P_ratio
PIXELS = (512, 256, 128, 64, 32,)
NUM_REALIZATIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NEX_TOTAL = (2500,)
bounds_type = None
#
for a_nex_total in NEX_TOTAL:
    for a_real in NUM_REALIZATIONS:
        for a_pixel in PIXELS:
            #
            RVE_MIA = mpi_cnn.load_data_set(
                os.path.join(DATA_SETS_DIR, f"rve_iss_{a_pixel}x{a_pixel}_training_images.npy"),
                permute_dims=(0, 3, 1, 2)
            )
            RVE_META = mpi_cnn.load_data_set(os.path.join(DATA_SETS_DIR, f"rve_iss_training_labels.npy"))
            rve_mia = RVE_MIA[:a_nex_total, 0:1]
            rve_meta = RVE_META[:a_nex_total]
            #
            nex_train = int(a_nex_total * 0.8)
            nex_val = a_nex_total - nex_train
            #
            (Training_Data_Loader, Validation_Data_Loader) = mpi_cnn.make_data_loaders(
                (rve_mia, rve_meta[:, 5:8], rve_meta[:, 0:1], rve_meta[:, 1:2], rve_meta[:, 3:4],),
                BATCH_SIZE,
                (nex_train, nex_val),
                shuffle=True
            )
            #
            model_id = f"NEX_{a_nex_total}-R_{a_real}-P_{a_pixel}"
            print(f"\n{'*' * 50}\nWorking on {model_id}\n{'*' * 50}")
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
                bounds_type=bounds_type
            ).to(my_device)
            MPI_CNN.init_weights(method="xavier_uniform")
            if PRINT_MODEL_SUMMARY:
                mpi_cnn.summary(MPI_CNN)
            MPI_CNN.trainer(
                train_dl=Training_Data_Loader,
                optimizer=mpi_cnn.torch.optim.Adam(MPI_CNN.parameters(), lr=0.001),
                num_epochs=NUM_EPOCHS,
                val_dl=Validation_Data_Loader,
                test_dl=None,
            )
            del MPI_CNN  # deleting the model from memory.
