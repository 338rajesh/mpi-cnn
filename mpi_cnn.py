import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchinfo import summary

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({"font.size": 20.0})


def make_data_loaders(xy: tuple[torch.Tensor], b_size, fractions=None, shuffle=True):
    xy = TensorDataset(*xy)
    if fractions is not None:
        if isinstance(fractions, (list, tuple)):
            return [
                DataLoader(a_xy, batch_size=b_size, shuffle=shuffle)
                for a_xy in random_split(xy, fractions)
            ]
        else:
            return DataLoader(xy, batch_size=b_size, shuffle=shuffle)
    else:
        return DataLoader(xy, batch_size=b_size, shuffle=shuffle)


def load_data_set(f_path: str, permute_dims: tuple = None, merge_chunks: bool = True) -> torch.Tensor:
    ds = np.load(f_path)  # MIA: Material_Information_Arrays
    if merge_chunks:
        ds = np.reshape(ds, newshape=(-1,) + ds.shape[2:])  # merging 0&1 axes
    ds = torch.tensor(data=ds, dtype=torch.float32, device=my_device)
    if permute_dims is not None:
        ds = torch.permute(ds, dims=permute_dims)
    return ds


def load_model(model_path, inc, ouu, els_mod_min, els_mod_max, p_ratio, print_summary=False, bounds_type=None):
    model_ = MpiCNN(
        in_channels=inc, out_units=ouu, e_min=els_mod_min, e_max=els_mod_max, nu=p_ratio, bounds_type=bounds_type
    ).to(my_device)
    model_.load_state_dict(torch.load(model_path))
    model_.eval()
    summary(model_) if print_summary else None
    return model_


class MpiCNN(nn.Module):

    @staticmethod
    def a_conv_op(inc, ouc, padding=(1, 1)):
        return nn.Sequential(
            nn.Conv2d(inc, ouc, kernel_size=(3, 3),
                      stride=(1, 1), padding=padding, ),
            nn.ReLU(inplace=True),
        )

    def __init__(
            self, in_channels, out_units, e_min, e_max, nu,
            working_dir=None, bounds_type=None, verbose=0,
    ):
        super(MpiCNN, self).__init__()
        self.work_dir = os.getcwd() if working_dir is None else working_dir
        self.metrics = None
        self.min_els_mod = e_min
        self.max_els_mod = e_max
        self.p_ratio = nu
        self.bounds_type = bounds_type
        self.verbose = verbose
        #
        self.conv_00 = self.a_conv_op(in_channels, 8)
        self.conv_01 = self.a_conv_op(8, 8)
        self.conv_02 = self.a_conv_op(8, 8)
        #
        self.conv_10 = self.a_conv_op(8, 16)
        self.conv_11 = self.a_conv_op(16, 16)
        self.conv_12 = self.a_conv_op(16, 16)
        #
        self.conv_20 = self.a_conv_op(16, 32)
        self.conv_21 = self.a_conv_op(32, 32)
        #
        self.avg_pooling = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        #
        self.flatten = nn.Flatten()
        #
        self.dense_layer_0 = nn.Sequential(nn.Linear(32, 32), nn.ReLU(inplace=True))
        self.outer_layer = nn.Linear(32, out_units)
        self.outer_activation = nn.Tanh()

    def init_weights(self, layer_types=(nn.Linear, nn.Conv2d), method="xavier_uniform"):
        if isinstance(self, layer_types):
            if method == "xavier_uniform":
                torch.nn.init.xavier_uniform_(self.weight)
                self.bias.data.fill_(0.01)
            else:
                raise ValueError(f"Invalid Initializer Type: {method}")

    def forward(self, images_batch, vf, em, ef):
        #
        # MIA preparation
        j = torch.ones_like(images_batch)
        em = torch.reshape(em, (em.shape[0], 1, 1, 1))
        ef = torch.reshape(ef, (ef.shape[0], 1, 1, 1))
        mia = (((em - self.min_els_mod) * j) + ((ef - em) * images_batch)) / (self.max_els_mod - self.min_els_mod)
        x = torch.concat(tensors=(images_batch, mia), dim=1)  # concatenating along the channel dimension
        #
        # VGG block_1
        x = self.conv_00(x)
        x = self.conv_01(x)
        x = self.conv_02(x)
        x = self.avg_pooling(x)
        #
        # VGG block_2
        x = self.conv_10(x)
        x = self.conv_11(x)
        x = self.conv_12(x)
        x = self.avg_pooling(x)
        #
        # VGG block_3
        x = self.conv_20(x)
        x = self.conv_21(x)
        x = self.avg_pooling(x)
        #
        # Global Average Pooling
        x = torch.mean(x, dim=(2, 3))
        #
        # Dense Layers and Output layer
        x = self.dense_layer_0(x)
        x = self.outer_layer(x)
        if self.bounds_type in ("VRH", "HS"):
            x = self.outer_activation(x)  # mapping bounds to [-1, 1] range
            #
            # evaluating and scaling to bounds
            lb, ub = self.eval_bounds(vf.squeeze(), em.squeeze(), ef.squeeze(), self.bounds_type)
            x = lb + (((1 + x) / 2) * (ub - lb))
        #
        return x

    @staticmethod
    def eval_vrh_bounds(vf, em, ef):
        """
            Returns a tuple of normalised lower and upper bound tensors of (n, 3) shape.
        """
        ecr = ef / em
        normalised_ub = (ecr * vf) + (1 - vf)
        normalised_lb = ecr / (vf + (ecr * (1.0 - vf)))
        return (
            torch.stack([normalised_lb for _ in range(3)], dim=1).squeeze(),
            torch.stack([normalised_ub for _ in range(3)], dim=1).squeeze()
        )

    @staticmethod
    def eval_hs_bounds(vf, em, num, ef, nuf):
        """
            Returns a tuple of normalised lower and upper bound tensors of (n, 3) shape.
        """

        def bulk_modulus(els_mod, p_rat):
            return els_mod / (3.0 * (1 - (2.0 * p_rat)))

        def shear_modulus(els_mod, p_rat):
            return els_mod / (2.0 * (1 + p_rat))

        def elastic_modulus(blk_mod, shr_mod):
            return (9.0 * blk_mod * shr_mod) / ((3.0 * blk_mod) + shr_mod)

        #
        vm = 1.0 - vf
        km, kf = bulk_modulus(em, num), bulk_modulus(ef, nuf)
        gm, gf = shear_modulus(em, num), shear_modulus(ef, nuf)
        #
        a = 1.0 / (kf - km)
        b = 1.0 / (gf - gm)
        c = vm / (km + gm)
        d = vf / (kf + gf)
        e = 1.0 + (km / (2.0 * gm))
        f = 1.0 + (kf / (2.0 * gf))
        #
        #
        k_lb = km + (vf / (a + c))
        k_ub = kf - (vm / (a - d))
        g_lb = gm + (vf / (b + (e * c)))
        g_ub = gf - (vm / (b - (f * d)))
        e_lb = elastic_modulus(k_lb, g_lb)  # (9.0 * k_lb * g_lb) / ((3.0 * k_lb) + g_lb)
        e_ub = elastic_modulus(k_ub, g_ub)  # (9.0 * k_ub * g_ub) / ((3.0 * k_ub) + g_ub)
        e_lb, e_ub = e_lb / em, e_ub / em
        g_lb, g_ub = g_lb / gm, g_ub / gm
        return (
            torch.stack((e_lb, e_lb, g_lb), dim=1).squeeze(),
            torch.stack((e_ub, e_ub, g_ub), dim=1).squeeze(),
        )

    def eval_bounds(self, vf, em, ef, bounds_type):
        if bounds_type in ("VRH", "HS", "NOB"):
            if bounds_type == "VRH":
                return self.eval_vrh_bounds(vf, em, ef)
            elif bounds_type == "HS":
                return self.eval_hs_bounds(vf, em, self.p_ratio, ef, self.p_ratio)
        else:
            raise ValueError(f"Invalid bounds type: {self.bounds_type}")

    def eval_mse_loss(self, predictions, truth, vf, em, ef, lbr_wt=1.0, ubr_wt=1.0):
        prediction_loss = nn.MSELoss(reduction="mean")(predictions, truth)
        return prediction_loss

    @staticmethod
    def eval_mape(predictions, truth):
        return (torch.mean(torch.abs(1.0 - (predictions / truth))) * 100.0).detach().to("cpu")

    @staticmethod
    def eval_ape(predictions, truth):
        return (torch.abs(1.0 - (predictions / truth)) * 100.0).detach().to("cpu")

    def predictions_on(self, data_loader: DataLoader, ) -> torch.Tensor:
        with torch.no_grad():
            predictions = torch.zeros(size=(0, 3,), device=my_device)
            for (rve_img, eff_prop, vf, em, ef) in data_loader:
                predictions = torch.concat(
                    tensors=(predictions, self.forward(rve_img, vf, em, ef)),
                    dim=0,
                )
            return predictions

    def eval_ape_on(self, data_loader: DataLoader):
        with torch.no_grad():
            ape_all = torch.zeros(size=(0, 3,), device=my_device)
            for (rve_img, eff_prop, vf, em, ef) in data_loader:
                ape_ = torch.abs(1.0 - (self.forward(rve_img, vf, em, ef) / eff_prop)) * 100.0
                ape_all = torch.concat(tensors=(ape_all, ape_), dim=0)
        return ape_all.cpu()

    def eval_mape_on(self, data_loader: DataLoader, along_dim=None):
        with torch.no_grad():
            ape_ = self.eval_ape_on(data_loader)
            mape = torch.mean(ape_, dim=along_dim)
        return mape

    def outliers_to_bounds_on(self, data_loader: DataLoader, bounds_type=None):
        with torch.no_grad():
            lb_outliers = torch.zeros(size=(0,), device=my_device)
            ub_outliers = torch.zeros(size=(0,), device=my_device)
            for (rve_img, eff_prop, vf, em, ef) in data_loader:
                prd = self.forward(rve_img, vf, em, ef).ravel()
                if bounds_type == "VRH":
                    lb, ub = self.eval_vrh_bounds(vf, em, ef)
                elif bounds_type == "HS":
                    lb, ub = self.eval_hs_bounds(vf, em, self.p_ratio, ef, self.p_ratio)
                lb = lb.ravel()
                ub = ub.ravel()
                lb_outliers = torch.concat((lb_outliers, prd[prd < lb]), dim=0)
                ub_outliers = torch.concat((ub_outliers, prd[prd > ub]), dim=0)
        return lb_outliers, ub_outliers

    def save_metrics(self):
        metrics_dir = os.path.join(self.work_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        #
        metrics = {k: np.array(v) for (k, v) in self.metrics.items()}
        np.savez_compressed(os.path.join(
            metrics_dir, "metrics.npz"), **metrics)
        for (k, v) in metrics.items():
            if k == "model-level":
                if v.size > 0:
                    with open(os.path.join(metrics_dir, "model-level-metrics.txt"), "w") as fh:
                        fh.write(f"Model Training Time: {v[0]}")
                        fh.write(f"Test set MAPE: {v[1]}")
            else:
                if k == "epoch-level":
                    header = f"{'Epoch':^10s}{'Train.MAPE':^20s}{'Val.MAPE':^20s}{'Time (in sec)':^20s}"
                elif k == "iter-level":
                    header = f"{'Iteration':^10s}{'Train.Loss (MSE)':^20s}{'Time (in sec)':^20s}"
                else:
                    raise KeyError(f"Invalid key {k} in the metrics.")
                #
                with open(os.path.join(metrics_dir, f"{k}-metrics.txt"), "w") as fh:
                    fh.write("_" * len(header) + f"\n{header}\n" + "_" * len(header) + "\n")
                    for a_row in v:
                        print_values = []
                        for (k1, i) in enumerate(a_row):
                            if k1 == 0:
                                print_values.append(f"{int(i):^10d}")
                            else:
                                print_values.append(f"{i:^20.3f}")
                        #
                        fh.write("".join(print_values) + "\n")
            #
        return

    def trainer(self, train_dl, optimizer, num_epochs: int = 10, val_dl=None, test_dl=None, save_hook="val"):
        self.metrics: dict = {"iter-level": [], "epoch-level": [], "model-level": []}
        best_mape, best_val_mape = float('inf'), float('inf')
        iter_count = 1
        description = f"@Epoch 1"
        model_t0 = time.time()
        #
        # Beginning of Training: Run over each epoch
        for a_epoch in range(num_epochs):
            a_epoch_metrics = [a_epoch + 1]
            a_epoch_t0 = time.time()
            tq = tqdm(leave=False, total=len(train_dl), desc=description)
            #
            for (rve_img, eff_prop, vf, em, ef) in train_dl:  # iterating on each batch of the training-set
                a_iter_metrics = [iter_count]
                a_iter_t0 = time.time()
                iter_count += 1
                #
                # LEARNING STEP
                optimizer.zero_grad()
                prediction = self.forward(rve_img, vf, em, ef, )
                loss = self.eval_mse_loss(prediction, eff_prop, vf, em, ef, lbr_wt=1.0, ubr_wt=1.0)
                loss.backward()
                optimizer.step()
                #
                a_iter_metrics.extend([loss.item(), time.time() - a_iter_t0])
                # iter-level metrics update
                self.metrics["iter-level"].append(a_iter_metrics)
                tq.update()
            del prediction, loss
            #
            # Evaluate performance metrics: MAPE on train set (at the end of epoch)
            train_mape = self.eval_mape_on(train_dl)
            a_epoch_metrics.append(train_mape)
            description = f"@Epoch {a_epoch}: tr-MAPE: {train_mape:4.3} || @Epoch {a_epoch + 1}: "
            #
            # Evaluating on the validation data set
            if save_hook == "train":
                if train_mape < best_mape:
                    best_mape = train_mape
                    torch.save(self.state_dict(), os.path.join(self.work_dir, "best_model.pth"))
                a_epoch_metrics.append(-10000.0)  # meaning no validation mape
            elif save_hook == "val":
                assert val_dl is not None, "Saving criteria is set to validation performance but dataset is not found."
                val_mape = self.eval_mape_on(val_dl)
                a_epoch_metrics.append(val_mape)
                description = (f"@Epoch {a_epoch}: tr-MAPE={train_mape:4.3} & val-MAPE={val_mape:4.3f}"
                               f" ||||| @Epoch {a_epoch + 1}: ")
                # saving the model at best mape of validation set
                if val_mape < best_val_mape:
                    best_val_mape = val_mape
                    torch.save(self.state_dict(), os.path.join(self.work_dir, "best_model.pth"))
            else:
                raise Warning(f"Invalid save_hook {save_hook} is found.")
            #
            a_epoch_metrics.append(time.time() - a_epoch_t0)
            self.metrics["epoch-level"].append(a_epoch_metrics)
            if a_epoch % 10 == 0:
                self.save_metrics()  # saving metrics to disc after every epoch,
            tq.close()
        # End of Training
        print(f"At the end of training with {num_epochs} epochs:")
        if save_hook == "train":
            print(f"\tBest TrainingSet-MAPE: {best_mape:4.3f}")
        elif save_hook == "val":
            print(f"\tBest ValidationSet-MAPE: {best_val_mape:4.3f}")
        #
        if test_dl is not None:  # Performance evaluation on test data Set
            eff_properties, predictions = self.eval_mape_on(test_dl)
            test_set_mape = self.eval_mape(predictions, eff_properties)
            self.metrics["model-level"].append(time.time() - model_t0)
            self.metrics["model-level"].append(test_set_mape)
            print(f"\tBest TestSet-MAPE: {test_set_mape}")
        #
        self.save_metrics()
        return self
