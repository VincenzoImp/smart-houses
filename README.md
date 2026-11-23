# Smart Houses

Smart Houses is a 2022 university project on demand response in a home energy management system. In an hour-ahead electricity market the customer only knows the price for the current hour. The project forecasts the next 12 hourly prices with an LSTM network and then uses reinforcement learning to decide, hour by hour, how much to charge a plug-in electric vehicle (PEV) battery, trading off energy cost against user dissatisfaction.

Project page: https://vincenzo.imperati.dev/projects/smart-houses

## How it works

The project has two phases, each with its own entry point.

**Phase 1: data preparation and price forecasting** (`main.py`, `DataPreprocessing.py`, `LongShortTermMemory.py`)

- `datas/energy.60.csv` holds hourly market prices for 2014. Missing prices are filled with a 2-nearest-neighbour imputer.
- Each home's 5-minute profile (`profiles.csv`) is aggregated into hourly rows (`new_profiles.csv`): price, consumption, PV production, PEV state of charge when plugged in, and the number of hours the PEV stays connected.
- `NN_datas.csv` is built from sliding windows over the price series. Each row holds the 50 previous hourly prices (input) and the following prices; the model predicts the next 12.
- Two Keras models are trained on the first 70% of the rows (no shuffling) and tested on the last 30%:
  - a baseline: two 50-unit LSTM layers with dropout and a 12-unit dense output;
  - a model tuned with Keras Tuner Hyperband (number of LSTM layers and units, dropout, dense activation, learning rate), minimising RMSE.
- Test-set predictions and true values are written, unscaled, to `NN_baseline_results.csv` and `NN_hypermodel_results.csv` (columns `predictions01..12` and `reals01..12`). Loss and prediction plots go to `datas/plot/`.

**Phase 2: battery charging simulation** (`RL/`)

- `RL/main.py` simulates selected homes in parallel processes. For each hour of the LSTM test period, the price vector is the current price plus the 12 hypermodel predictions.
- Each charging strategy is a device configured from a CSV file in the home's folder:
  - `NSL_Battery` (`NSLpev.csv`): charges at full speed as soon as the vehicle is plugged in. It is the reference for evaluation.
  - `Naif_Battery` (`Naifpev.csv`): fills the hours with the lowest known prices first, within the time the vehicle stays connected. `deficit` is added to the current state of charge when planning, so a higher value plans less charging.
  - `CL_Battery` (`CLpev.csv`): tabular Q-learning (`pyqlearning` `GreedyQLearning`). The state is the step within the price horizon and the discretised state of charge (16 levels); actions are discrete charging amounts between 0 and the charge speed (3.6 kW). The reward is `1 - y/d`, where `y = (1 - p) * price * kWh + p * beta * (kWh - max)^2` combines cost and dissatisfaction. Every hour the agent learns for 1,000 loops before choosing an action.
- Home parameters (`RL/Home.py`): `p` weights dissatisfaction against cost; learning rate 0.1, discount 0.95, epsilon 0.2.
- `RL/Evaluation.py` compares each device with `NSL_Battery.0` as percentage differences in energy cost, average price per kWh, kWh charged and average state of charge at unplugging, and combines them into a score weighted by `p`.

The shiftable-load (`SL*`), generic controllable-load (`CL.py`) and dynamic-programming (`DP_Battery.py`) devices are present but incomplete or disabled in `Simulation.insert_devices`.

## Repository layout

- `main.py`, `DataPreprocessing.py`, `LongShortTermMemory.py`: phase 1.
- `RL/`: phase 2 (simulation, devices, Q-learning models, evaluation).
- `datas/`: price data, neural-network datasets and results, and plots in `datas/plot/`.
- `datas/muratori_5/`: 62 home folders, each with a 5-minute `profiles.csv` and the hourly `new_profiles.csv`. Only `home_69` also has the device CSV files used by the simulation.
- `datas/simulations/`: four simulation runs of `home_69`. Each holds `main.csv` (hourly totals), one CSV per device, `home_info.csv`, `device_info.csv` and `evaluation.csv`. The `CL_Battery.N_plots` folders contain one PNG per decision hour showing the Q-value over the learning loops; these 2,970 images are most of the files in the repository.
- `output/`: meant to hold the Keras Tuner trials. They were left out because of their size (see `output/README.txt`).
- `Smart Houses.pdf`: the project report.

## Running

There is no requirements file. The code imports pandas, numpy, matplotlib, scikit-learn, tensorflow/keras, keras-tuner and pyqlearning.

```sh
# Phase 1, from the repository root
python main.py          # full dataset
python main.py True     # test mode: samples 30% of the dataset

# Phase 2, from inside RL/ (paths are relative to that folder)
cd RL
python main.py
```

Phase 1 rewrites `datas/energy.60.csv`, the `new_profiles.csv` files and the result CSVs and plots. Because the tuner trials are not included, the hyperparameter search runs again from scratch. Phase 2 simulates the homes listed in `houses_to_simulate` (by default `home_69`, with `p = 0.2`) and writes a new timestamped folder in `datas/simulations/`.

## Results

- Forecasting plots: [baseline loss](datas/plot/baseline_loss.svg), [baseline predictions](datas/plot/baseline_model.svg), [tuned model loss](datas/plot/hypermodel_loss.svg), [tuned model predictions](datas/plot/hyper_model.svg), [prices and consumption over one week](datas/plot/prices_and_consumptions.svg).
- Simulation results for `home_69` are in `datas/simulations/*/evaluation.csv`. Three runs compare nine Q-learning agents (different `beta`) and three naive strategies with `p` set to 0.2, 0.5 and 0.8; a fourth run compares five naive strategies only. In the three Q-learning runs every agent pays less for energy than the reference, but charges fewer kWh and leaves the battery at a lower state of charge.
- The report, [Smart Houses.pdf](Smart%20Houses.pdf), describes the method and results in full.

## Authors

Deborah Dore ([deborahdore](https://github.com/deborahdore)) and Vincenzo Imperati ([VincenzoImp](https://github.com/VincenzoImp)).

## Licence

MIT. See [LICENSE](LICENSE).
