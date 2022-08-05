"""
CLI that evaluates a given study using the given metrics
"""
import re
import os
import sys
import json
import argparse
from typing import Dict, List, Tuple
from glob import glob
from datetime import datetime, timezone

import SimpleITK as sitk
import numpy as np
import pandas as pd


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# pylint: disable=wrong-import-position
# Fix path
from eval.metrics import get_metric

REGEX_STRING = (
    ".*pred_(?P<experiment>.*)s2_user_(?P<user>\d+)_volume_(?P<volume>\d+)_"
    + "(?P<date>\d{4}-\d{2}-\d{2}-\d{6}).nii.gz"
)


def load_file_as_array(file_path: str, seed_value: int) -> Tuple[np.ndarray, List[int]]:
    """
    Loads a given .nii.gz file as array

    :param file_path: Path to file that should be loaded
    :type file_path: str
    :param seed_value: Value in the file that is interpreted as foreground.
    :type seed_value: int
    :return: Tuple (array, spacing). 1 in the array marks the foreground, 0 the background
    :rtype: Tuple[np.ndarray, List[int]]
    """
    assert os.path.exists(file_path), f"Could not find file {file_path}!"

    img = sitk.ReadImage(file_path)
    img_arr = sitk.GetArrayFromImage(img)
    spacing = list(img.GetSpacing())

    if np.all(seed_value != img_arr):
        # TODO hacky
        # Default to 1
        seed_value = 1

    binary_array = np.where(img_arr == seed_value, 1, 0)

    return binary_array, spacing


def evaluate_files(
    label_path: str,
    folder_paths: List[str],
    metrics_list: List[str] = None,
    label_seed_value: int = 1,
    annotation_seed_value: int = 2,
    creation_times: Dict[str, float] = None,
) -> pd.DataFrame:
    """
    Evaluates the annotations in the folders using the given label as basis

    :param label_path: Path to the label .nii.gz file
    :type label_path: str
    :param folder_paths: List of folders that contain .nii.gz files. Each folder is interpreted as \
        one experiment/user
    :type folder_paths: List[str]
    :param metrics_list: List of metrics that should be calculated on the data, defaults to ["dice"]
    :type metrics_list: List[str], optional
    :param label_seed_value: Value that marks the foreground in the label, defaults to 1
    :type label_seed_value: int, optional
    :param annotation_seed_value: Value that marks the foreground in the annotations, defaults to 2
    :type annotation_seed_value: int, optional
    :return: Metric dataframe. Has the columns "name", "spacing", "ctime", "time", "score", \
        "full_path", "experiment", "index", "metric" & "label"
    :rtype: pd.DataFrame
    """

    assert os.path.exists(label_path), f"'label_path' does not exist: {label_path}"
    assert len(folder_paths) > 0, "Missing parameter 'folder_paths'"
    for path in folder_paths:
        assert os.path.exists(path), f"Entry in 'folder_paths' does not exist: {path}"
        assert os.path.isdir(path), f"{path} is not a directory!"

    if metrics_list is None or len(metrics_list) == 0:
        metrics_list = ["dice"]

    if creation_times is None:
        creation_times = {}

    # Load label
    ground_truth, _ = load_file_as_array(label_path, label_seed_value)

    metric_func_list = []

    # Resolve metrics
    for metric in metrics_list:
        if metric is None or len(metric) <= 0:
            continue
        try:
            metric_func_list.append(get_metric(metric_name=metric))
        except ValueError as e:
            metric_func_list.append(None)
            print(f"Could not resolve metric '{metric}': {e}")

    file_meta = []

    # TODO if spacing is different-> might have to resample
    for current_experiment in folder_paths:

        annotation_files = sorted(
            glob(os.path.join(current_experiment, "*.nii.gz")), key=os.path.getctime
        )

        experiment_name = current_experiment
        if experiment_name[-1] == "\\" or experiment_name[-1] == "/":
            experiment_name = experiment_name[:-1]
        experiment_name = os.path.basename(experiment_name)

        study_base_time = None
        for index, current_file in enumerate(annotation_files):
            current_arr, current_spacing = load_file_as_array(
                current_file, annotation_seed_value
            )

            file_match = re.match(REGEX_STRING, current_file)
            uses_gc = False

            if file_match is None:
                current_time = os.path.getctime(current_file)
            else:
                current_time = (
                    datetime.strptime(file_match.group("date"), "%Y-%m-%d-%H%M%S")
                    .replace(tzinfo=timezone.utc)
                    .astimezone(None)
                    .timestamp()
                )

                uses_gc = "gc" in file_match.group("experiment")

            if study_base_time is None:
                study_base_time = creation_times.get(current_experiment, current_time)

            for func_ind, func in enumerate(metric_func_list):
                if func is None:
                    continue

                file_meta.append(
                    {
                        "name": os.path.basename(current_file),
                        "spacing": current_spacing,
                        "ctime": current_time,
                        "time": current_time - study_base_time,
                        "score": func(current_arr, ground_truth),
                        "full_path": current_file,
                        "experiment": experiment_name,
                        "index": index,
                        "metric": metric_list[func_ind],
                        "label": label_path,
                        "gc": uses_gc,
                    }
                )

    metrics_df = pd.DataFrame.from_records(file_meta)
    return metrics_df


def evaluate_dict(
    study_paths: Dict[str, Dict[str, List[str]]],
    metrics_list: List[str] = None,
    label_seed_value: int = 1,
    annotation_seed_value: int = 2,
) -> pd.DataFrame:
    """
    Iterates through the input dictionary, evaluates each setting and combines them into one \
    dataframe

    :param study_paths: Dictionary containing the study settings. Keys: labels, Values: List of \
        folders that contain the user annotations
    :type study_paths: Dict[str, List[str]]
    :param metrics_list: List of metrics that should be calculated on the data, defaults to ["dice"]
    :type metrics_list: List[str], optional
    :param label_seed_value: Value that marks the foreground in the label, defaults to 1
    :type label_seed_value: int, optional
    :param annotation_seed_value: Value that marks the foreground in the annotations, defaults to 2
    :type annotation_seed_value: int, optional
    :return: Metric dataframe. Has the columns "name", "spacing", "ctime", "time", "score", \
        "full_path", "experiment", "index", "metric" & "label"
    :rtype: pd.DataFrame
    """
    if metrics_list is None or len(metrics_list) == 0:
        metrics_list = ["dice"]

    resulting_df_list = []
    for label_path, folder_dict in study_paths.items():
        print(f"Evaluating {label_path}")

        # Parse creation times
        creation_times_dict = {
            folder: datetime.strptime(creation_list[0], "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=timezone.utc)
            .astimezone(None)
            .timestamp()  # TODO timestamp
            for folder, creation_list in folder_dict.items()
        }
        resulting_df_list.append(
            evaluate_files(
                label_path,
                folder_dict.keys(),
                metrics_list=metrics_list,
                label_seed_value=label_seed_value,
                annotation_seed_value=annotation_seed_value,
                creation_times=creation_times_dict,
            )
        )

    return pd.concat(resulting_df_list).sort_values(by="ctime")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "file",
        metavar="config-file-path",
        type=str,
        help="Path to a JSON file that contains the labels and annotation paths. The json"
        + " contains the paths to the labels. Example: {'label_1_path': {"
        + "'annotation_folder_path': ['2022-07-30 10:15:00']}}.",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        type=str,
        default="dice",
        help="Metric(s) that will be evaluated, defaults to 'dice'. Multiple metrics can be"
        + " chained with a semicolon, i.e. 'dice;mse'. Possible options are: 'dice'",
    )

    parser.add_argument(
        "--no-dash",
        action="store_true",
        help="Flag, if present the results will not be rendered using dash.",
    )

    parser.add_argument("--debug", action="store_true", help="Activates debug mode.")

    parser.add_argument(
        "--csv-path",
        type=str,
        default=".",
        help="Folder where the resulting csv should be saved to. The csv name is "
        + "'eval_result_<time>.csv",
    )

    parser.add_argument(
        "--no-save-csv",
        action="store_true",
        help="Deactivates saving of resulting csv file. During debug mode the file can only be "
        + "saved when --no-dash is present due to dash's hot reloading function.",
    )

    parser.add_argument("--port", type=int, default=8050, help="Dash app port.")
    parser.add_argument(
        "--label-seed-value",
        type=int,
        default=1,
        help="Value that marks foreground in the label.",
    )
    parser.add_argument(
        "--annotation-seed-value",
        type=int,
        default=1,
        help="Value that marks foreground in the annotation.",
    )

    args = parser.parse_args()

    metric_list = args.metrics.split(";")

    # read json file
    with open(args.file, encoding="utf-8") as setup_file:
        study_setup = json.load(setup_file)

    print(f"Using set-up file at {args.file}")
    resulting_scores = evaluate_dict(
        study_setup,
        metric_list,
        label_seed_value=args.label_seed_value,
        annotation_seed_value=args.annotation_seed_value,
    )

    if not args.no_save_csv:
        if not args.debug or args.no_dash:
            c_time = datetime.now().strftime("%Y%m%d%H%M%S")
            result_file_path = os.path.join(args.csv_path, f"eval_result{c_time}.csv")

            print(f"Saving result to {result_file_path}.")
            resulting_scores.to_csv(result_file_path, index=False)
        else:
            print(
                "Skipping saving since the app was started in debug mode, in order to avoid "
                + "overload due to dash's hot reloading!"
            )

    if args.no_dash:
        print(resulting_scores)
    else:
        from eval.visualize_evaluation import create_evalutaion_app

        app = create_evalutaion_app(resulting_scores)
        app.run_server(debug=args.debug, port=args.port)
