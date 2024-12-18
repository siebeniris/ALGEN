import json
import os
import re

import pandas as pd
import plac


data = {
    "cosine_similarities": {
        "X_Y_COS": 0.6784769892692566,
        "X_Y_test_COS": 0.557990550994873,
        "X_Y_COS_LOSS": 0.3215230107307434,
        "X_Y_MSE_LOSS": 461022068736.0,
        "X_Y_test_COS_LOSS": 0.44200941920280457,
        "X_Y_test_MSE_LOSS": 32362919936.0
    },
    "rouge_results": {
        "X_vs_gold": {
            "rouge1_f": 0.049397164924801465,
            "rouge2_f": 0.00034493012191495683,
            "rougeL_f": 0.04819024555791253
        },
        "Y_vs_gold": {
            "rouge1_f": 0.0016609147248746803,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0016609147248746803
        },
        "X_vs_Y": {
            "rouge1_f": 0.00020408163265306123,
            "rouge2_f": 0.0,
            "rougeL_f": 0.00020408163265306123
        }
    }
}

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_one_row(filename, data_dir, train_samples):
    # Use regex to find the number following 'reg'
    reg_match = re.search(r"reg(\d+\.\d+)", filename)
    # Use regex to find the number following 'regm'
    regm_match = re.search(r"regm(\d+\.\d+)", filename)

    if reg_match and regm_match:
        reg_value = float(reg_match.group(1))
        regm_value = float(regm_match.group(1))
        print(reg_value, regm_value)

        with open(os.path.join(data_dir, filename)) as f:
            result_dict = json.load(f)

        flattend = flatten_dict(result_dict)
        flattend["reg"] = reg_value
        flattend["regm"] = regm_value
        flattend["train"] = train_samples
        return pd.DataFrame(flattend)
    else:
        print("no regm and reg found")

def get_result_for_one_model(data_dir):

    rows = []
    for subdir in os.listdir(data_dir):

        dirpath = os.path.join(data_dir, subdir)
        if os.path.isdir(dirpath):
            match = re.search(r"_samples_(\d+)", subdir)
            if match:
                samples_number = int(match.group(1))

                for filename in os.listdir(dirpath):
                    if filename.endswith(".json"):
                        row = get_one_row(filename, dirpath, samples_number)
                        rows.append(row)
    df = pd.concat(rows, axis=0, ignore_index=True)

    df.to_csv(f"{data_dir}/results.csv",index=False)


if __name__ == '__main__':
    # get_one_row("ub_sinkhorn_reg0.03_regm0.009_eval_results.json")
    # d = flatten_dict(data)
    # d["reg"]=0.1
    # d["reg_m"]=0.1
    # print(d)
    # df = pd.DataFrame([d])
    # print(df.columns)

    plac.call(get_result_for_one_model)

