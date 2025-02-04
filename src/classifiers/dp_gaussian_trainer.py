from src.classifiers.trainer import fine_tune


source_models = ["google-t5/t5-base", "google/mt5-base",
                      "sentence-transformers/gtr-t5-base",
                      "google-bert/bert-base-multilingual-cased"]

datasets_names = ["yiyic/snli_ds", "yiyic/sst2_ds", "yiyic/s140_ds"]
tasks = ["sentiment", "nli"]

# epsilon_delta_l = [(0.01, 0.001), (0.5, 0.0001), (0.01, 0.0001)]

epsilon_delta_l = [(1.0, 1e-06),
                   (0.1, 1e-06),
                   (0.05, 0.0001),
                   (0.01, 0.001),
                   (0.5, 0.0001),
                   (0.5, 0.001),
                   (0.1, 0.0001),
                   (0.05, 1e-05),
                   (1.0, 0.0001),
                   (0.01, 1e-05),
                   (0.05, 1e-05),
                   (0.01, 0.0001),
                   (0.05, 0.0001)]


def dp_gaussian_trainer():
    for model_name in source_models:
        for dataset_name in datasets_names:
            for e_d in epsilon_delta_l:
                epsilon, delta = e_d
                if dataset_name == "yiyic/snli_ds":
                    task = "nli"
                    num_labels = 3
                else:
                    task = "sentiment"
                    num_labels =2

                fine_tune(dataset_name, task, num_labels, model_name,
                          128, "dp_Gaussian",
                          0, delta, epsilon)


if __name__ == '__main__':
    dp_gaussian_trainer()
