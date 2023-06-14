## Structure of the datasets

Data sets can be accessed on [Kaggle](https://doi.org/10.34740/kaggle/ds/3402384) or [Zeono](https://doi.org/10.5281/zenodo.8035643)

The data set contains the transverse elastic properties of uni-directional composite materials
with the following details.

### Datasets: D1, D2, D3 and D4

|                  | Dataset: D1            | Dataset: D2           | Dataset: D3            | Dataset: D4            |
|----------------- |------------------------|-----------------------|------------------------|------------------------|
| **vf** range     | [25, 75]               | [25, 75]              | [10, 25]               | [10, 25]               |
| **Ecr** range    | [5, 250]               | [250, 500]            | [250, 500]             | [5, 250]               |
| **X_train** shape| (80, 250, 256, 256, 1) |     -                 |       -                |       -                |
| **y_train** shape| (80, 250, 8)           |     -                 |       -                |       -                |
| **X_test** shape | (40, 250, 256, 256, 1) | (20, 250, 256, 256, 1)| ( 6, 250, 256, 256, 1) | (15, 200, 256, 256, 1) |
| **y_test** shape | (40, 250, 8)           | (20, 250, 8)          | ( 6, 250, 8)           | (15, 200, 8)           |

Here,
+ vf: fibre volume fraction
+ Ecr: Elastic moduli contrast, the ratio of fibre elastic moduls to that of the matrix
+ Each of the label arrays contain 8 columns, which are
	+   column index 	=> 		label
		`0 				=>    Vf`, Fibre volume fraction
		`1 				=>    Em`, Matrix elastic modulus
		`2 				=>    NUm`, Matrix Poisson's ratio
		`3 				=>    Ef`, Fibre elastic modulus
		`4 				=>    NUf`, Fibre Poisson's ratio
		`5 				=>    E22/Em`
		`6 				=>    E33/Em`
		`7 				=>    G23/Gm`

