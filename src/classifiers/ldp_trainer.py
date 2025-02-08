from src.classifiers.trainer import fine_tune

source_models = ["google-t5/t5-base", "google/mt5-base",
                 "sentence-transformers/gtr-t5-base",
                 "google-bert/bert-base-multilingual-cased",
                 "text-embedding-ada-002"]

datasets_names = ["yiyic/snli_ds", "yiyic/sst2_ds", "yiyic/s140_ds"]

tasks = ["sentiment", "nli"]

defenses = ["PurMech", "LapMech"]
epsilon_l = [1, 2, 4, 6, 8, 10, 12]


def ldp_trainer(dataset_name):
    for model_name in source_models:
        for defense in defenses:
            for epsilon in epsilon_l:

                if dataset_name == "yiyic/snli_ds":
                    task = "nli"
                    num_labels = 3
                else:
                    task = "sentiment"
                    num_labels = 2

                fine_tune(dataset_name, task, num_labels, model_name,
                          128, defense,
                          0, 0, epsilon)


if __name__ == '__main__':
    import plac

    plac.call(ldp_trainer)
