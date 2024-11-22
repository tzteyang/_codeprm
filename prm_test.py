import json
import fire
import os
import torch
import copy
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from prm_utils import get_process_rewards, CODEPRM_PROMPT


class PRMTest:
    def __init__(self, 
                 model_name_or_path: str = None,
                 data_file: str = None,
                 output_dir: str = None,
                 run_seed: int = 42,):
        self.model_name_or_path = model_name_or_path
        self.data_file = data_file
        self.output_dir = output_dir
        self.seed = run_seed
        if model_name_or_path is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            # breakpoint()
    
    def collect_data(self, output_file: str = None, max_samples_per_question: int = 10):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            # data = [json.loads(line) for line in f.readlines()]
            data = json.load(f)
        with open(self.data_file.replace('raw_data2', 'raw_data'), 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        # breakpoint()
        collected_data = []
        for question, question_annotation in data.items():
            if question in data2:
                continue
            prompt = CODEPRM_PROMPT.format(question=question)
            _temp_data = []
            random.seed(self.seed)
            question_annotation = random.sample(question_annotation, min(max_samples_per_question, len(question_annotation)))
            for solution_steps in question_annotation:
                if solution_steps.get("has_final_step", None) is None:
                    continue
                response_steps = solution_steps["solution_prefix"].removeprefix('### Solution').strip()
                # label = "positive" if solution_steps["mc_value"] >= threshold else "negative"
                _temp_data.append({
                    "prompt": prompt,
                    "response": response_steps,
                    # "has_final_step": solution_steps["has_final_step"],
                    "mc_value": solution_steps["mc_value"],
                })
            collected_data.extend(_temp_data)
        
        if output_file is not None and not os.path.exists(output_file):
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(collected_data, ensure_ascii=False, indent=4))
            return None
        
        return collected_data
    
    def run(self,
            batch_size: int = 8,):

        collected_data = self.collect_data(max_samples_per_question=50)
        # breakpoint()
        output_fn, output_ex = os.path.splitext(os.path.basename(self.data_file))
        output_fn += '_prm_scored'
        # breakpoint()
        output_file = os.path.join(self.output_dir, output_fn + output_ex)
        start_idx = 0
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                scored_data = [json.loads(line) for line in f.readlines()]
            start_idx = len(scored_data)
        
        # for idx, data in enumerate(tqdm(collected_data[start_idx:], desc='Process annotating...'), start=start_idx):

        for batch_idx in tqdm(range(start_idx, len(collected_data), batch_size), desc='Process annotating...'):
            batch_collected = collected_data[batch_idx:batch_idx+batch_size]
            prompts = [collected["prompt"] for collected in batch_collected]
            process_responsess = [[collected_data["response"]] for collected_data in batch_collected]
            judge_scores = get_process_rewards(
                self.model,
                self.tokenizer,
                prompts=prompts,
                completed_processes=process_responsess,
                tokenized_format='chat_completion',
                reward_strategy='token_logits'
            ).cpu().tolist()
            # breakpoint()
            run_results = copy.deepcopy(batch_collected)
            for idx, judge_score in enumerate(judge_scores):
                run_results[idx]["judge_score"] = judge_score
            with open(output_file, 'a', encoding='utf-8') as f:
                for res in run_results:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')
            torch.cuda.empty_cache()
            # breakpoint()
    
    def analyze_score_distributions(self,
                                    analyze_file: str,
                                    save_fig_file: str = None,
                                    threshold: float = 0.5):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
        from scipy.stats import gaussian_kde

        with open(analyze_file, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]

        scores = np.array([item['judge_score'][0] for item in data])
        mc_values = np.array([item['mc_value'] for item in data])
        
        plt.figure(figsize=(8, 6))
        bins = np.linspace(min(min(mc_values), min(scores)), 
                        max(max(mc_values), max(scores)), 31)
        
        plt.hist(mc_values, bins=bins, alpha=0.4, label='MC Values', 
                color='blue', edgecolor='black')
        plt.hist(scores, bins=bins, alpha=0.4, label='Judge Scores[p(`+`)]', 
                color='red', edgecolor='black')
        
        plt.title('Distribution Comparison')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        
        if save_fig_file is not None:
            plt.savefig(save_fig_file.replace('.png', '_dist.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        
        xy = np.vstack([mc_values, scores])
        density = gaussian_kde(xy)(xy)
        
        plt.scatter(mc_values, scores, c=density, 
                   cmap='OrRd', alpha=1.0)
        plt.colorbar(label='Density')
        
        plt.axhline(y=threshold, color='r', linestyle='--', 
                   label=f'Threshold={threshold}')
        plt.axvline(x=threshold, color='r', linestyle='--')
        
        z = np.polyfit(mc_values, scores, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(mc_values), max(mc_values), 100)
        plt.plot(x_trend, p(x_trend), "g--", 
                label=f'Trend line (slope={z[0]:.3f})')
        
        plt.title('Score vs MC Value')
        plt.xlabel('MC Value')
        plt.ylabel('Judge Score[0]')
        plt.legend()
        
        predicted = (scores >= threshold).astype(int)
        actual = (mc_values >= threshold).astype(int)
        accuracy = np.mean(predicted == actual)
        
        tp = np.sum((predicted == 1) & (actual == 1))  # True Positive
        tn = np.sum((predicted == 0) & (actual == 0))  # True Negative
        fp = np.sum((predicted == 1) & (actual == 0))  # False Positive
        fn = np.sum((predicted == 0) & (actual == 1))  # False Negative
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        stats_text = (
            f'Statistical Metrics:\n\n'
            f'Samples: {len(scores)}\n'
            f'Correlation: {np.corrcoef(mc_values, scores)[0,1]:.3f}\n'
            f'Accuracy: {accuracy:.3f}\n'
            f'Precision: {precision:.3f}\n'
            f'Recall: {recall:.3f}\n'
            f'F1 Score: {f1:.3f}\n\n'
            f'Confusion Matrix:\n'
            f'TP: {tp}  FP: {fp}\n'
            f'FN: {fn}  TN: {tn}'
        )
        
        plt.text(-0.3, 0.5, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='center',
                bbox=dict(boxstyle='round',
                         facecolor='white',
                         alpha=0.8))
        
        # plt.tight_layout()
        plt.subplots_adjust(right=0.8)
        
        if save_fig_file is not None:
            plt.savefig(save_fig_file.replace('.png', '_scatter.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
        # print("\nCompute Metrics:")
        # print(f"Accuracy: {accuracy:.3f}")
        # print(f"Precision: {precision:.3f}")
        # print(f"Recall: {recall:.3f}")
        # print(f"F1 Score: {f1:.3f}")
        
        # print("\nConfusion Matrix:")
        # print(f"True Positive: {tp}")
        # print(f"True Negative: {tn}")
        # print(f"False Positive: {fp}")
        # print(f"False Negative: {fn}")
        
        return None


if __name__ == '__main__':
    fire.Fire(PRMTest)    