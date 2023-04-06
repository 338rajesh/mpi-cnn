## Structure of the datasets

### Datasets: D1, D2, D3 and D4

|                  | Dataset: D1            | Dataset: D2           | Dataset: D3            | Dataset: D4            |
|----------------- |------------------------|-----------------------|------------------------|------------------------|
| **vf** range     | [25, 75]               | [25, 75]              | [10, 25]               | [10, 25]               |
| **Ecr** range    | [5, 250]               | [250, 500]            | [250, 500]             | [5, 250]               |
| **X_train** shape| (80, 250, 256, 256, 1) |     -                 |       -                |       -                |
| **y_train** shape| (80, 250, 8)           |     -                 |       -                |       -                |
| **X_test** shape | (40, 250, 256, 256, 1) | (20, 250, 256, 256, 1)| ( 6, 250, 256, 256, 1) | (15, 200, 256, 256, 1) |
| **y_test** shape | (40, 250, 8)           | (20, 250, 8)          | ( 6, 250, 8)           | (15, 200, 8)           |

### Labels

column index => label

1. `0 =>    Vf`, Fibre volume fraction
2. `1 =>    Em`, Matrix elastic modulus
3. `2 =>    NUm`, Matrix Poisson's ratio
4. `3 =>    Ef`, Fibre elastic modulus
5. `4 =>    NUf`, Fibre Poisson's ratio
6. `5 =>    E22/Em`
7. `6 =>    E33/Em`
8. `7 =>    G23/Gm`
