import os
import numpy as np

CSD = os.path.dirname(__file__)

# ==============================================

BOUNDS_TYPES = ("NOB", "HS")
DATA_SET_IDS = ("NATIVE", "EXT_DS2", "EXT_DS3", "EXT_DS4", "PLB")
NEX_DS = tuple(f'{i}' for i in (5000, 5000, 1500, 1500, 10000))


def eval_normalised_hs_bounds(vf, em, num, ef, nuf):
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
        np.concatenate([e_lb, e_lb, g_lb], axis=1),
        np.concatenate([e_ub, e_ub, g_ub], axis=1),
    )


f = open(os.path.join(CSD, "outliers_summary.txt"), "w")

for a_real in range(10):
    f.write(f"\n\n{'*'*80}\n{'REALISATION: '+str(a_real):^80s}\n{'*'*80}\n")
    for (i, a_data_set_id) in enumerate(DATA_SET_IDS):

        f.write(f"\n{a_data_set_id+' data set with size='+NEX_DS[i]:<35s}\n{'-'*35}\n")
        f.write(
            f"{'Model Trainined on':^20s}"
            # f"|{'# of outliers to VR-bounds of':^36s}"
            f"|{'# of outliers to HS-bounds of':^60s}"
            f"|\n"
        )
        # |
        f.write(f"{'':^20s}|{'E22':^20s}{'E33':^20s}{'G23':^20s}|\n\n")
        for a_bound_type in BOUNDS_TYPES:
            test_res = np.load(os.path.join(CSD, f"Test-Results-{a_data_set_id}", f"test_results-{a_bound_type}.npz"))
            model_id = f"NEX_10000-R_{a_real}"
            if a_bound_type == "VR":
                model_id += f"-B_VRH"
            elif a_bound_type == "HS":
                model_id += f"-B_HS"
            #
            if model_id in test_res.keys():
                model_predictions = test_res[model_id]
            else:
                raise KeyError(f"{model_id} is not found in {list(test_res.keys())}")
            vf, em, ef = model_predictions[:, 0:1], model_predictions[:, 1:2], model_predictions[:, 3:4]
            true_values = model_predictions[:, 5:8]
            predictions = model_predictions[:, 8:11]
            #
            # Find the number of outliers to VR bounds
            # vr_lb, vr_ub = eval_normalised_vrh_bounds(vf, ecr=ef / em)
            # vrl_outliers = [predictions[predictions[:, i] < vr_lb[:, i]] for i in range(3)]
            # vru_outliers = [predictions[predictions[:, i] > vr_ub[:, i]] for i in range(3)]
            # no_vrl = [i.size for i in vrl_outliers]
            #
            # Find the number of outliers to HS bounds
            hs_lb, hs_ub = eval_normalised_hs_bounds(vf, em, 0.25, ef, 0.25)
            hsl_outliers = [predictions[predictions[:, i] < hs_lb[:, i]] for i in range(3)]
            hsu_outliers = [predictions[predictions[:, i] > hs_ub[:, i]] for i in range(3)]
            no2_e22, no2_e33, no2_g23 = [i.shape[0] for i in hsl_outliers]
            f.write(
                f"{a_bound_type:^20s}"
                f"{no2_e22:^20d}"
                f"{no2_e33:^20d}"
                f"{no2_g23:^20d}"
                f"\n\n"
            )
