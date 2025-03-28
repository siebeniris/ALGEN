from src.classifiers.trainer import fine_tune

source_models = ["google-t5/t5-base", "google/mt5-base",
                 "sentence-transformers/gtr-t5-base",
                 "google-bert/bert-base-multilingual-cased",
                 "text-embedding-ada-002"]

# datasets_names = ["yiyic/snli_ds", "yiyic/sst2_ds", "yiyic/s140_ds"]
# datasets_names = [ "yiyic/snli_ds"]
tasks = ["sentiment", "nli"]

noise_levels = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]


def gaussian_trainer(dataset_name):
    for model_name in source_models:
        for noise_level in noise_levels:
            if dataset_name == "yiyic/snli_ds":
                task = "nli"
                num_labels = 3
            else:
                task = "sentiment"
                num_labels = 2

            fine_tune(dataset_name, task, num_labels, model_name,
                      128, "Gaussian",
                      noise_level, 0, 0)


if __name__ == '__main__':
    import plac

    plac.call(gaussian_trainer)
