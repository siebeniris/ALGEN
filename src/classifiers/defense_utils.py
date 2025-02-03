
def dp_gaussian_params():
    # epsilon and delta, selected for the extreme values.
    return [(0.01, 0.001), (0.5, 0.0001), (0.01, 0.0001)]


def gaussian_noise():

    return [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]




source_models = ["google-t5/t5-base", "google/mt5-base",
                      "sentence-transformers/gtr-t5-base",
                      "google-bert/bert-base-multilingual-cased"]


