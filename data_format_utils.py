import os
import re
import json
import fire
from typing import Tuple, Dict, Optional
from pathlib import Path
from process_data_annotation.prompts import CODEPRM_PROMPT
from process_data_annotation.checker_utils import CodeSolutionParser


class CodeOmegaPRM:
    """CodeOmega PRM data collection and analysis tool."""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path.cwd()
    
    def collect(self, 
                data_dir: Optional[str] = None, 
                output_file: Optional[str] = None,
                problem_pattern: str = r'problem_(\d+)\.json') -> Tuple[Dict, Dict]:
        working_dir = Path(data_dir) if data_dir else self.data_dir
        
        if not working_dir.exists():
            raise ValueError(f"Directory not found: {working_dir}")
            
        existing_problems_results = [
            f for f in working_dir.glob('*.json')
            if re.match(problem_pattern, f.name)
        ]
        
        collected_tree_data = {}
        collected_steps_data = {}
        
        for problem_file in existing_problems_results:
            try:
                with problem_file.open('r', encoding='utf-8') as f:
                    problem_data = json.load(f)

                question = problem_data.pop("question")
                collected_steps_data[question] = problem_data.get("steps_data")
                collected_tree_data[question] = problem_data.get("tree_data")
            except Exception as e:
                import traceback
                print(f"Error processing {problem_file}: {traceback.format_exc()}")
        
        if output_file:
            output_path = Path(output_file)
            output_data = {
                "steps_data": collected_steps_data,
                "tree_data": collected_tree_data
            }
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
                print(f"Results saved to {output_path}")
            
            return None
        
        return collected_steps_data, collected_tree_data
    
    def analyze_value_distribution(self, 
                data_path: Optional[str] = None,
                output_file: Optional[str] = None) -> Dict:
        """
        Analyze the collected data and generate statistics and visualizations.
        
        Args:
            data_dir: Optional directory path containing the data files
            analysis_type: Type of analysis to perform ('basic' by default)
            output_file: Optional file path to save the analysis results
            
        Returns:
            Dict containing analysis results
        """
        # First collect the data
        # steps_data, tree_data = self.collect(data_dir)

        if data_path is not None:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            steps_data = data['steps_data']
            tree_data = data['tree_data']
        
        if not steps_data:
            raise ValueError("No data found to analyze")
        
        # Extract all mc_values
        mc_values = []
        for question_data in steps_data.values():
            if isinstance(question_data, list):
                for step in question_data:
                    if isinstance(step, dict) and 'mc_value' in step:
                        mc_values.append(step['mc_value'])
        
        if not mc_values:
            raise ValueError("No MC values found in the data")
        
        # Calculate statistics
        import numpy as np
        import matplotlib.pyplot as plt
        
         # Calculate statistics
        mean_value = np.mean(mc_values)
        median_value = np.median(mc_values)
        
        # Create visualization
        plt.figure(figsize=(12, 7))
        
        n, bins, patches = plt.hist(mc_values, bins=20, edgecolor='black')
        
        for i in range(len(patches)):
            if n[i] > 0:  
                plt.text(
                    patches[i].get_x() + patches[i].get_width()/2.,
                    patches[i].get_height(), 
                    f'{int(n[i])}',
                    ha='center',
                    va='bottom'
                )
        
        plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.3f}')
        plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.3f}')
        plt.xlabel('MC Values')
        plt.ylabel('Frequency')
        plt.title('Distribution of MC Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization if output file is specified
        if output_file:
            output_path = Path(output_file)
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        
        # Prepare analysis results
        results = {
            'statistics': {
                'mean': float(mean_value),
                'median': float(median_value),
                'total_values': len(mc_values),
                'min': float(np.min(mc_values)),
                'max': float(np.max(mc_values)),
                'std': float(np.std(mc_values))
            },
            # 'distribution': {
            #     'histogram': np.histogram(mc_values, bins=20),
            #     'raw_values': mc_values
            # }
        }
        
        return results
    
    def analyze_steps_info(self,
                           data_path: Optional[str] = None,
                           preprocess: bool = True,
                           output_file: Optional[str] = None) -> Dict:
        if preprocess:
            if data_path is not None:
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                steps_data = data['steps_data']
                tree_data = data['tree_data']
            code_solution_parser = CodeSolutionParser()
            for question, question_annotation in steps_data.items():
                for solution_step in question_annotation:
                    solution_prefix = solution_step.get("solution_prefix", "")
                    if not solution_prefix:
                        continue
                    parsed_info = code_solution_parser.process_solution(solution_prefix)
                    steps_count = parsed_info["total_steps"]
                    has_final_step = parsed_info["has_code_generation"]
                    solution_step.update({
                        "steps_count": steps_count,
                        "has_final_step": has_final_step
                    })

            if output_file:
                output_path = Path(output_file)
                # breakpoint()
                with output_path.open('w', encoding='utf-8') as f:
                    json.dump(steps_data, f, indent=4, ensure_ascii=False)
                    print(f"Results saved to {output_path}")
        else:
            # analyze the saved data
            import matplotlib.pyplot as plt
            import numpy as np
            
            with open(data_path, 'r', encoding='utf-8') as f:
                steps_data = json.load(f)
            
            steps_counts = []
            has_final_counts = {'True': 0, 'False': 0}
            
            for question_data in steps_data.values():
                for step in question_data:
                    if isinstance(step, dict):
                        if 'steps_count' in step:
                            steps_counts.append(step['steps_count'])
                        
                        if 'has_final_step' in step:
                            has_final_counts[str(step['has_final_step'])] += 1
            
            plt.figure(figsize=(15, 6))
            
            plt.style.use('bmh')  # 使用 matplotlib 内置的 bmh 样式
        
            plt.figure(figsize=(15, 6), facecolor='white')
            
            main_color = '#6c5ce7'  # 柱状图主色
            grid_color = '#bdc3c7'  # 网格线颜色
            text_color = '#2c3e50'  # 文字颜色
            pie_colors = ['#a8e6cf', '#ff8b94']  # 饼图颜色（薄荷绿和粉红色）
            
            plt.subplot(121)
            n, bins, patches = plt.hist(steps_counts, 
                                    bins=max(5, min(20, len(set(steps_counts)))), 
                                    color=main_color,
                                    edgecolor='white',
                                    alpha=0.7)
            
            for i in range(len(patches)):
                if n[i] > 0:  # 只在有值的柱子上添加标签
                    plt.text(
                        patches[i].get_x() + patches[i].get_width()/2.,
                        patches[i].get_height(),
                        f'{int(n[i])}',
                        ha='center',
                        va='bottom',
                        color=text_color,
                        fontweight='bold'
                    )
            
            plt.title('Distribution of Steps Count', color=text_color, fontsize=12, pad=15)
            plt.xlabel('Number of Steps', color=text_color)
            plt.ylabel('Frequency', color=text_color)
            plt.grid(True, alpha=0.3, color=grid_color, linestyle='--')
            
            plt.tick_params(colors=text_color)
            
            # 2. has_final_step 饼图
            plt.subplot(122)
            patches, texts, autotexts = plt.pie(
                [has_final_counts['True'], has_final_counts['False']], 
                labels=[f'Has Final Step\n({has_final_counts["True"]})',
                    f'No Final Step\n({has_final_counts["False"]})'],
                autopct='%1.1f%%',
                colors=pie_colors,
                textprops={'color': text_color},
                wedgeprops={'alpha': 0.8, 'edgecolor': 'white'},
                startangle=90
            )
            
            for autotext in autotexts:
                autotext.set_color(text_color)
                autotext.set_fontweight('bold')
            
            plt.title('Distribution of Solutions with Final Step', color=text_color, fontsize=12, pad=15)
            
            plt.tight_layout()
            plt.title('Distribution of Solutions with Final Step')
            
            plt.tight_layout()
            
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Analysis results saved to: {output_path}")
            
            stats = {
                'steps_count': {
                    'mean': float(np.mean(steps_counts)),
                    'median': float(np.median(steps_counts)),
                    'min': min(steps_counts),
                    'max': max(steps_counts),
                    'total_samples': len(steps_counts)
                },
                'has_final_step': {
                    'with_final': has_final_counts['True'],
                    'without_final': has_final_counts['False'],
                    'percentage_with_final': has_final_counts['True'] / sum(has_final_counts.values()) * 100
                }
            }
            
            print("\nAnalysis Summary:")
            print(f"Steps Count Statistics:")
            print(f"  Total samples: {stats['steps_count']['total_samples']}")
            print(f"  Mean steps: {stats['steps_count']['mean']:.2f}")
            print(f"  Median steps: {stats['steps_count']['median']:.2f}")
            print(f"  Range: [{stats['steps_count']['min']}, {stats['steps_count']['max']}]")
            print(f"\nFinal Step Statistics:")
            print(f"  Solutions with final step: {stats['has_final_step']['with_final']} "
                f"({stats['has_final_step']['percentage_with_final']:.1f}%)")
            print(f"  Solutions without final step: {stats['has_final_step']['without_final']}")
            
            return None

    def to_prm_train_format(self,
                            data_path: Optional[str],
                            output_file: Optional[str] = None,
                            threshold: float = 0.5) -> None:
        with open(data_path, 'r', encoding='utf-8') as f:
            steps_data = json.load(f)
        
        prm_raw_data = []
        for question, question_annotation in steps_data.items():
            prompt = CODEPRM_PROMPT.format(question=question)
            _temp_data = []
            for solution_steps in question_annotation:
                if solution_steps.get("has_final_step", None) is None:
                    continue
                response_steps = solution_steps["solution_prefix"]
                label = "positive" if solution_steps["mc_value"] >= threshold else "negative"
                _temp_data.append({
                    "prompt": prompt,
                    "response": response_steps,
                    "has_final_step": solution_steps["has_final_step"],
                    "label": [label]
                })
            prm_raw_data.extend(_temp_data)
        
        if output_file:
            output_path = Path(output_file)
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(prm_raw_data, f, indent=4, ensure_ascii=False)
                print(f"Results saved to {output_path}")
    

    def info(self) -> Dict:
        ...

def main():
    fire.Fire(CodeOmegaPRM) 

if __name__ == '__main__':
    main()