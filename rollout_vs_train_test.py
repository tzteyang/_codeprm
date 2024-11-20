import re
import glob
import json
import os
import fire
import ast
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Any, Union, Optional
from functools import partial


def data_analyze(directory_path, start_range=0, end_range=500):
    def _collect_target_data(directory_path, start_range=0, end_range=500):
        collected_data = []

        train_pattern = re.compile(f"train_(\d+)-(\d+)\.json")
        # pattern = re.compile(f"train_(\d+)-(\d+)\.json$")
        
        all_files = glob.glob(os.path.join(directory_path, "*.json"))
        
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            match = train_pattern.search(file_name)
            if match:
                file_start = int(match.group(1))
                file_end = int(match.group(2))
                if file_start >= start_range and file_end <= end_range:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = [json.loads(line) for line in f.readlines()]
                        temp_data = {
                            'file_name': file_name,
                            'range': (file_start, file_end),
                            'train_data': data
                        }

                        train_result_file_name, extension = os.path.splitext(file_name)
                        train_result_file_name += "_result"
                        if not os.path.exists(file_path.replace(file_name, train_result_file_name + extension)):
                            continue
                        with open(file_path.replace(file_name, train_result_file_name + extension), 'r', encoding='utf-8') as f:
                            data = json.load(f)["scores"]
                        temp_data.update({"train_result_data": data})

                        tested_file_name, extension = os.path.splitext(file_name)
                        tested_file_name += "_rollout_test"
                        if not os.path.exists(file_path.replace(file_name, tested_file_name + extension)):
                            continue
                        with open(file_path.replace(file_name, tested_file_name + extension), 'r', encoding='utf-8') as f:
                            data = [json.loads(line) for line in f.readlines()]
                        temp_data.update({"tested_data": data})

                        collected_data.append(temp_data)
                        print(f"read successfully: {file_path}")
                    except Exception as e:
                        print(f"when processing file '{file_path}' some error occurred: {e}")
        return collected_data

    x_label_pass_rates = []
    y_model_pass_rates = [] 
    collected_data = _collect_target_data(directory_path, start_range, end_range)
    for data_block in collected_data:
        train_result_data = data_block['train_result_data']
        tested_data = data_block['tested_data']
        # breakpoint()
        for idx in range(min(len(train_result_data), len(tested_data))):
            for label_completion, test_completion_rate in zip(tested_data[idx]["completions_k_samples"], train_result_data[idx]):

                x_label_pass_rates.append(sum(label_completion) / len(label_completion))
                y_model_pass_rates.append(test_completion_rate)

    # plt.figure(figsize=(8, 6))
    # plt.scatter(x_label_pass_rates, y_model_pass_rates, alpha=0.6)
    # plt.xlabel('Pass Rate of Synthetic Label Code')
    # plt.ylabel('Pass Rate of Model on Test Tasks')
    # plt.title('Relationship Between Synthetic Data Quality and Model Performance')
    # plt.grid(True)
    # plt.savefig('label-quality_vs_test-metrics.png', dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.figure(figsize=(8, 6))
    # plt.hist2d(x_label_pass_rates, y_model_pass_rates, bins=50, cmap='Blues')
    # plt.colorbar(label='Number of Data Points')
    # plt.xlabel('Pass Rate of Synthetic Label Code')
    # plt.ylabel('Pass Rate of Model on Test Tasks')
    # plt.title('2D Histogram of Synthetic Data Quality vs Model Performance')
    # plt.savefig('label-quality_vs_test-metrics.png', dpi=300, bbox_inches='tight')
    # plt.show()
    # import numpy as np
    # from scipy.stats import gaussian_kde

    # # Calculate the density of points
    # xy = np.vstack([x_label_pass_rates, y_model_pass_rates])
    # z = gaussian_kde(xy)(xy)

    # # Plot scatter plot with color representing density
    # plt.figure(figsize=(8, 6))
    # plt.scatter(x_label_pass_rates, y_model_pass_rates, c=z, s=50, cmap='Blues')
    # plt.xlabel('Pass Rate of Synthetic Label Code')
    # plt.ylabel('Pass Rate of Model on Test Tasks')
    # plt.title('Density Scatter Plot of Synthetic Data Quality vs Model Performance')
    # plt.colorbar(label='Density')
    # plt.savefig('label-quality_vs_test-metrics.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.figure(figsize=(8, 6))
    plt.hist(x_label_pass_rates, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Pass Rate of Synthetic Label Code')
    plt.ylabel('Frequency')
    plt.title('Histogram of Pass Rate Distribution for Synthetic Label Code')
    plt.grid(False)
    plt.savefig('label-quality_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    fire.Fire(data_analyze)