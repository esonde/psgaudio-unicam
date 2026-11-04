# Sleep Apnea Detection

This project tests whether reconstructing physiological signals from breathing sounds helps a model learn useful features for apnea detection. I compare a classifier attached to a frozen, regression-pretrained backbone with the same architecture trained directly from random initialization. Both ambient and tracheal recordings are considered.

Run the notebooks in order, with this folder as the working directory:

1. `1_download.ipynb` downloads PSG-Audio, extracts the recordings and physiological signals, and estimates the audio alignment. Set `DATA` to the folder where the raw data should be stored.
2. `2_preprocessing.ipynb` creates nonoverlapping 60-second windows, PCEN images, normalized physiological targets, validity masks, labels, and patient-wise splits. Set `DATA` to the same raw-data folder and `OUT` to the dataset folder.
3. `3_training.ipynb` trains and evaluates both approaches on three splits. Set `DATA` to the preprocessed dataset and `OUT` to the results folder. The architecture is defined in `model.py`.

The notebooks use Python, Jupyter, NumPy, SciPy, pandas, Matplotlib, scikit-learn, PyEDFlib, PyTorch, and torchaudio.

For tracheal audio, use `audio_tracheal` instead of `audio` in the preprocessing notebook and save to `dataset_tracheal`. Then train with `DATA = Path("dataset_tracheal")` and `OUT = Path("training_tracheal")`. Use the same `splits.json` for both microphones when comparing them; the currently saved dataset folders have different split files, and the final tracheal runs used the ambient splits.

Results are saved under `split2026`, `split2027`, and `split2028`, with separate `frozen_backbone` and `from_scratch` folders. They include the selected models, loss histories, classification predictions, and final metrics. Completed strategies are skipped when `results.json` exists. An interrupted training phase restarts from the beginning.

A classification window is positive when it overlaps annotated apnea for at least 10 seconds, and negative when it contains neither apnea nor hypopnea. Other windows are excluded from classification. Physiological signals are only needed during pretraining; inference uses audio alone.

`images.ipynb` generates the report figures. The report source is in `report/report.tex`.
