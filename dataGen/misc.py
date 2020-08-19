from tqdm import tqdm
import copy
import json
import os


def correct_val_splits():
    """
    Nitin mistakenly switched input-output samples. Script to correct the dataset
    :return:
    """
    json_files = [f for f in os.listdir("./Val_Curated") if f.endswith('.json')]
    for _json in json_files:
        num_samples = sum([1 for _ in open(os.path.join("./Val_Curated", _json), 'r')])
        pbar = tqdm(total=num_samples)
        corrected_data = []
        # Read data
        with open(os.path.join("./Val_Curated", _json), 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                corrected_data.append({
                    "input": data["output"],
                    "output": data["input"],
                    "program": data["program"]
                })
                pbar.update(1)
        pbar.close()
        # Write data
        with open(os.path.join("./Val_Curated", "new" + _json), 'w') as f_t:
            for index, sample in enumerate(corrected_data):
                if index == 0:
                    f_t.write(json.dumps(sample))
                else:
                    f_t.write('\n' + json.dumps(sample))


def combine_multiple_train_datasets(save_after):
    """
    Combine multiple same-sized datasets into one.
    :param save_after: Size of each dataset
    """

    # Path to the json that will contain the combined dataset
    combined_json = "train_no_test_large_half_m1.json"
    m = 1  # Number of examples to add from input-output dictionary
    num_datasets_to_combine = 5

    all_samples = []
    file_names = ["Train_10mil/train{}.json".format(id + 1) for id in range(num_datasets_to_combine)]
    pbar = tqdm(total=save_after * len(file_names))
    for _file in file_names:
        with open(_file, 'r') as f2:
            for line in f2:
                _data = json.loads(line)
                edited_data = {
                    "input": _data["input"][:m],
                    "output": _data["output"][:m],
                    "program": _data["program"]
                }
                all_samples.append(edited_data)
                pbar.update(1)

    pbar.close()
    with open(combined_json, 'w') as f1:
        for index, sample in enumerate(all_samples):
            if index == 0:
                f1.write(json.dumps(sample))
            else:
                f1.write('\n' + json.dumps(sample))


def measure_avg_prog_length(path_to_dataset_json):
    """
    Measure average length of transformation programs in the specified dataset
    """
    avg_length = 0
    max_seq_length = 40
    seq_with_len_less_than_max = 0
    total_samples = 0
    with open(path_to_dataset_json, 'r') as fr:
        for line in fr:
            total_samples += 1
            sample = json.loads(line)
            program_str = sample["program"]
            avg_length += len(program_str.split())
            if len(program_str.split()) <=  max_seq_length:
                seq_with_len_less_than_max += 1
    print("Avg Length: {}".format(avg_length/total_samples))
    print("Compatible Sequences: {}/{} = {}".format(seq_with_len_less_than_max, total_samples,
                                                    seq_with_len_less_than_max/total_samples))


def create_final_test_from_test_splits(src_dir="./Test_Curated/", target_path="./test.json"):
    """
    Collect specified number of samples from specified splits and combine them.
    :param src_dir: Path to directory containing files whose names are specified in test_split_from_val_split
    :param target_path: path to save the final json for test dataset
    :return:
    """
    # indicate for each file, how many samples to collect
    test_split_from_val_split = {
        # "modified_nonOverlap.json": 3524,
        # "modified_overlap.json": 4649,
        # "modified_partialOverlap.json": 1827,
        # "nonModified_nonOverlap.json": 4474,
        "nonModified_overlap.json": 3500,
        # "nonModified_partialOverlap.json": 2010,
        "onlyFunctionNames_nonOverlap.json": 6500,
        "onlyFunctionNames_overlap.json": 10000,
        # "onlyFunctionNames_partialOverlap.json": 1089
    }
    test_samples = []
    for file in test_split_from_val_split.keys():
        with open(os.path.join(src_dir, file), 'r') as fr:
            for index, line in enumerate(fr):
                data = json.loads(line)
                if data not in test_samples:
                    test_samples.append(data)

    print("Number of Test Samples: {}".format(len(test_samples)))
    with open(target_path, 'w') as fw:
        for index, sample in enumerate(test_samples):
            if index == 0:
                fw.write(json.dumps(sample))
            else:
                fw.write('\n' + json.dumps(sample))


def create_test_split_from_val_split(src_dir="./Val_Curated", trg_dir="./Test_Curated"):
    """
    To create a subset of validation splits
    :return:
    """
    # indicate for each file, how many samples to collect
    test_split_from_val_split = {
        "modified_nonOverlap.json": 3524,
        "modified_overlap.json": 4649,
        "modified_partialOverlap.json": 1827,
        "nonModified_nonOverlap.json": 4474,
        "nonModified_overlap.json": 3500,
        "nonModified_partialOverlap.json": 2010,
        "onlyFunctionNames_nonOverlap.json": 6500,
        "onlyFunctionNames_overlap.json": 10000,
        "onlyFunctionNames_partialOverlap.json": 1089
    }
    for file in test_split_from_val_split.keys():
        test_split_samples = []
        with open(os.path.join(src_dir, file), 'r') as fr:
            for index, line in enumerate(fr):
                if index < test_split_from_val_split[file]:
                    data = json.loads(line)
                    test_split_samples.append(data)
                else:
                    break
        with open(os.path.join(trg_dir, file), 'w') as fw:
            for index, sample in enumerate(test_split_samples):
                if index == 0:
                    fw.write(json.dumps(sample))
                else:
                    fw.write('\n' + json.dumps(sample))


def create_test_from_val(val_path, test_path):
    """
    Create test dataset (subset) from validation dataset
    :param val_path:
    :param test_path:
    :return:
    """
    test_samples = []
    num_test_samples = 20000
    with open(val_path, 'r') as fr:
        for index, line in enumerate(fr):
            test_dict = {}
            if index < num_test_samples:
                data = json.loads(line)
                test_dict['input'] = data['input']
                test_dict['output'] = data['output']
                test_dict['program'] = data['program']
                test_samples.append(test_dict)

    with open(test_path, 'w') as fw:
        for index, sample in enumerate(test_samples):
            if index == 0:
                fw.write(json.dumps(sample))
            else:
                fw.write('\n' + json.dumps(sample))


def collect_benchmark_tasks():
    """
    Collect csv files for benchmark tasks from original flashFill dataset as per the ids specified in bechmark_file
    :return:
    """
    num_digits = 6  # Number of digits in task_id corresponding to each Flash-fill CSV

    bechmark_file = "no_logic_tasks.json"
    source_dir = "FlashFill_orig_CSV"
    target_dir = "FlashFill_Curated_CSV"

    with open(bechmark_file, 'r') as f:
        file_details = json.load(f)

    file_names: List[str] = []
    for task, task_ids in file_details.items():
        for _id in task_ids:
            file_name = task + "." + (num_digits - len(str(_id)))*"0" + str(_id) + ".csv"
            file_names.append(file_name)
            shutil.copy(os.path.join(source_dir, file_name), target_dir)


def gen_splits(samples: List):
    counts_based_split = {}
    for sample_dict in samples:
        length = len(sample_dict["input"])
        if length not in counts_based_split.keys():
            counts_based_split[length] = [sample_dict]
        else:
            counts_based_split[length].extend([sample_dict])
    return counts_based_split


def collect_annotations_from_flashfill_csvs(flash_fill_src_dir="FlashFill_Curated_CSV",
                                            flash_fill_trg_dir="FlashFill_Curated_Json"):
    """
    :param flash_fill_src_dir: Directory to read csv files from
    :param flash_fill_trg_dir: Directory to save json files for flashfill tasks
    :return:
    """
    save_empty = False  # combined json file has empty output strings allowed for inputs strings
    save_non_empty = False  # combined json file does not have ip, op string pairs with empty output
    save_empty_split = False  # Multiple json files created based on number of ip-op strings per task
    # (with empty output strings allowed)
    save_non_empty_split = False  # Multiple json files created based on number of ip-op strings per task
    # (with no empty output strings allowed)

    fixed_samples = 10

    csv_files = [file for file in os.listdir(flash_fill_src_dir) if not file.startswith(".")]
    print("Total CSV files: {}".format(len(csv_files)))

    column_names = ["Input", "Output"]

    # We will maintain two separate Json, one that keeps inp-out pairs irrespective of whether
    # output is empty or non-empty and other Json containing only pairs with non-empty output
    non_empty_samples, empty_samples, fixed_num_samples = [], [], []
    for csv_file in csv_files:
        # print("File: {}".format(csv_file))
        non_empty_sample_dict = {
            "input": [],
            "output": [],
            "csv_file": str(csv_file)
        }
        empty_sample_dict = {
            "input": [],
            "output": [],
            "csv_file": str(csv_file)
        }
        csv_file_path = os.path.join(flash_fill_src_dir, csv_file)
        df = pd.read_csv(csv_file_path, usecols=column_names, na_filter=False)  # convert nan entries to empty strings

        for inp, out in zip(df["Input"], df["Output"]):
            empty_sample_dict["input"].append(str(inp))
            empty_sample_dict["output"].append(str(out))
            if inp and out:
                non_empty_sample_dict["input"].append(str(inp))
                non_empty_sample_dict["output"].append(str(out))

        non_empty_samples.append(non_empty_sample_dict)
        empty_samples.append(empty_sample_dict)

        # Check to display flashfill tasks with 'fixed_samples' number of minimum example pairs
        if len(non_empty_sample_dict["input"]) >= fixed_samples:
            print("Retained: {}".format(non_empty_sample_dict["csv_file"]))
            fixed_num_samples.append({
                "input": non_empty_sample_dict["input"][:fixed_samples],
                "output": non_empty_sample_dict["output"][:fixed_samples],
                "csv_file": non_empty_sample_dict["csv_file"]
            })

    if save_non_empty:
        with open(os.path.join(flash_fill_trg_dir, "flashFill_raw_no_empty_op.json"), 'w') as f:
            for index, sample in enumerate(non_empty_samples):
                if index == 0:
                    f.write(json.dumps(sample))
                else:
                    f.write('\n' + json.dumps(sample))

    if save_empty:
        with open(os.path.join(flash_fill_trg_dir, "flashFill_raw_with_empty_op.json"), 'w') as f:
            for index, sample in enumerate(empty_samples):
                if index == 0:
                    f.write(json.dumps(sample))
                else:
                    f.write('\n' + json.dumps(sample))

    if save_non_empty_split:
        count_based_split = gen_splits(non_empty_samples)
        for num_examples in count_based_split.keys():
            with open(os.path.join(flash_fill_trg_dir, "no_empty_op_json_splits/test_{}.json".format(num_examples)),
                      'w') as f:
                for index, sample in enumerate(count_based_split[num_examples]):
                    if index == 0:
                        f.write(json.dumps(sample))
                    else:
                        f.write('\n' + json.dumps(sample))

    if save_empty_split:
        count_based_split = gen_splits(empty_samples)
        for num_examples in count_based_split.keys():
            with open(os.path.join(flash_fill_trg_dir, "with_empty_op/test_{}.json".format(num_examples)),
                      'w') as f:
                for index, sample in enumerate(count_based_split[num_examples]):
                    if index == 0:
                        f.write(json.dumps(sample))
                    else:
                        f.write('\n' + json.dumps(sample))

    print("Total tasks to be retained so that each has exactly {} I, O pairs: {}".format(fixed_samples,
                                                                                         len(fixed_num_samples)))
    with open(os.path.join(flash_fill_trg_dir,"flashFill_10_samples_per_task.json"), 'w') as f:
        for index, sample in enumerate(fixed_num_samples):
            if index == 0:
                f.write(json.dumps(sample))
            else:
                f.write('\n' + json.dumps(sample))