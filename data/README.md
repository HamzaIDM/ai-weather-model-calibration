# Data

The raw dataset used in this project (`dataset_fusionnée.csv`, about 50 MB) was provided by the
General Directorate of Meteorology of Morocco (DGM) for the internship. **It is not distributed in this
repository.** To run the notebooks, place a file with the schema below at `data/dataset_fusionnée.csv`.

Each row pairs one AI-model forecast with the observation recorded at the same station and valid time.

| Column | Type | Description |
|---|---|---|
| `run_time` | date | Initialisation date of the AI model run |
| `lead_time` | int (h) | Forecast horizon, 0 to 72 hours |
| `model` | str | AI model: `aifs`, `aurora`, `dlwp`, `fengwu`, `fourcastnet`, `fuxi`, `graphcast`, `pangu6` |
| `station` | str | Station name (34 Moroccan synoptic stations) |
| `distance_km` | float | Distance between the station and the grid point used |
| `lat`, `lon` | float | Coordinates of the grid point |
| `variable_name` | str | `t2m` (°C), `td2m` (°C), `msl` (hPa), `sp` (hPa), `tp06` (mm / 6 h) |
| `pred_value` | float | Value forecast by the AI model |
| `valid_time` | datetime | `run_time + lead_time`; 00, 06, 12 or 18 UTC |
| `obs_value` | float | Value observed at the station (SYNOP) |
| `biais` | float | `pred_value − obs_value` |
| `erreur_abs` | float | Absolute error |
| `erreur_sq` | float | Squared error |

Coverage per model and variable (number of rows):

| Model | msl | sp | t2m | td2m | tp06 |
|---|---:|---:|---:|---:|---:|
| AIFS | 2,136 | 2,497 | 24,611 | 2,508 | 23,700 |
| Aurora | 1,880 | | 21,474 | | |
| DLWP | | | 21,474 | | |
| FengWu | 1,880 | | 21,474 | | |
| FourCastNet | 1,880 | 2,168 | 21,474 | | |
| FuXi | 1,880 | | 21,474 | | 20,932 |
| GraphCast | 1,880 | | 21,474 | | 20,932 |
| Pangu6 | 1,880 | | 21,474 | | |

Anything placed in this folder is ignored by git (see `.gitignore`).
