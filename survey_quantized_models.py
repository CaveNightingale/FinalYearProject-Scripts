#!/usr/bin/env python3
"""
抽样调查 HuggingFace 上量化 LLM 的配置信息
"""

import requests
import json
from typing import Dict, List, Optional
import time
from collections import Counter
import csv
import random
from datetime import datetime

class HFQuantizationSurvey:
    def __init__(self, sample_size: int = 100):
        self.sample_size = sample_size
        self.api_base = "https://huggingface.co/api"
        self.results = []
        
    def search_quantized_models(self) -> List[Dict]:
        """搜索可能的量化模型，并按各类别的总量进行配额采样"""
        print("正在搜索量化模型...")
        
        # 搜索关键词
        search_terms = [
            "GPTQ", "AWQ", "quantized", "int2",
            "int4", "int8", "4bit", "8bit",
            "fp4", "fp8", "nf4", "3bit", "2bit",
            "w8a8", "w4a16", "w4a8", "w4a4"
        ]
        
        # 第一阶段：使用 quicksearch API 获取每个类别的准确总数
        category_counts = {}  # {term: total_count}
        
        print("\n第一阶段：使用 quicksearch API 获取各类别的准确总数...")
        for term in search_terms:
            url = f"https://huggingface.co/api/quicksearch?q={term}&type=model"
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    count = data.get('modelsCount', 0)
                    category_counts[term] = count
                    print(f"  '{term}': {count:,} 个模型")
                else:
                    category_counts[term] = 0
                    print(f"  '{term}': 获取失败")
                time.sleep(0.3)
            except Exception as e:
                print(f"  '{term}' 出错: {e}")
                category_counts[term] = 0
        
        # 计算总数和各类别配额
        total_found = sum(category_counts.values())
        print(f"\n总模型数: {total_found:,} （可能有重复）")
        
        # 第二阶段：根据总数计算配额，并获取模型列表
        print("\n第二阶段：按比例分配采样配额并获取模型...")
        category_models = {}  # {term: [models]}
        all_models = []
        seen_ids = set()
        
        for term in search_terms:
            if category_counts[term] == 0:
                continue
            
            # 计算该类别应采样的数量（按比例）
            quota = int(self.sample_size * category_counts[term] / total_found)
            if quota == 0 and category_counts[term] > 0:
                quota = 1  # 至少采样1个
            
            fetch_count = quota * 3  # 多获取一些以备去重
            
            print(f"  '{term}': 总数 {category_counts[term]:,}, 配额 {quota}, 获取 {fetch_count}")
            
            # 获取该类别的模型（分页）
            term_models = []
            page = 0
            
            url = f"{self.api_base}/models"
            params = {
                "search": term,
                "filter": "text-generation",
                "sort": "downloads",
                "direction": -1,
                "limit": quota
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    models = response.json()
                    if not models:
                        break
                    term_models.extend(models)
                    if len(models) < quota:
                        break
                    page += 1
                    time.sleep(0.3)
                else:
                    break
            except Exception as e:
                print(f"    获取出错: {e}")
                break
            
            # 从获取的模型中采样
            sampled = 0
            for model in term_models:
                model_id = model.get('id') or model.get('modelId')
                if model_id and model_id not in seen_ids:
                    seen_ids.add(model_id)
                    all_models.append({
                        'model': model,
                        'search_term': term
                    })
                sampled += 1
            time.sleep(0.5)
        
        # 按下载量排序
        all_models.sort(key=lambda x: x['model'].get('downloads', 0), reverse=True)
        
        print(f"\n最终采样 {len(all_models)} 个不重复的模型")

        random.shuffle(all_models)

        return [m['model'] for m in all_models[:self.sample_size]]
    
    def get_model_config(self, model_id: str) -> Optional[Dict]:
        """获取模型的 config.json"""
        url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"    获取配置失败: {e}")
            return None
    
    def extract_quantization_info(self, config: Dict) -> Optional[Dict]:
        """提取量化配置信息"""
        quant_info = None
        
        # 检查 quantization_config
        if "quantization_config" in config:
            quant_info = config["quantization_config"]
        # 检查 quant_config
        elif "quant_config" in config:
            quant_info = config["quant_config"]
        elif "quantization" in config:
            quant_info = config["quantization"]
        
        return quant_info
    
    def survey_models(self):
        """调查模型配置"""
        models = self.search_quantized_models()
        
        print(f"\n开始调查 {len(models)} 个模型的配置...")
        print("=" * 80)
        
        for idx, model in enumerate(models, 1):
            model_id = model.get('id') or model.get('modelId')
            downloads = model.get('downloads', 0)
            
            print(f"\n[{idx}/{len(models)}] {model_id}")
            print(f"  下载量: {downloads:,}")
            
            # 获取配置
            config = self.get_model_config(model_id)
            
            if config is None:
                print("  ❌ 没有找到 config.json")
                continue
            
            # 提取量化配置
            quant_config = self.extract_quantization_info(config)
            
            if quant_config is None:
                print("  ⚠️  config.json 中没有量化配置字段，忽略处理")
            else:
                quant_method = quant_config.get('quant_method') or quant_config.get('quantization_method') or 'unknown'
                print(f"  ✅ 找到量化配置: {quant_method}")
                # print(f"     配置详情: {json.dumps(quant_config, indent=6)}")
                
                self.results.append({
                    'model_id': model_id,
                    'downloads': downloads,
                    'has_config': True,
                    'has_quant_config': True,
                    'quant_method': quant_method,
                    'quant_config': json.dumps(quant_config)
                })
            
            time.sleep(0.3)  # 避免请求过快
    
    def analyze_results(self):
        """分析调查结果"""
        print("\n" + "=" * 80)
        print("调查结果统计")
        print("=" * 80)
        
        total = len(self.results)
        has_config = sum(1 for r in self.results if r['has_config'])
        has_quant = sum(1 for r in self.results if r['has_quant_config'])
        
        print(f"\n总调查模型数: {total}")
        print(f"有 config.json 的模型: {has_config} ({has_config/total*100:.1f}%)")
        print(f"有量化配置的模型: {has_quant} ({has_quant/total*100:.1f}%)")
        
        # 统计量化方法
        print("\n量化方法分布:")
        quant_methods = [r['quant_method'] for r in self.results if r['quant_method']]
        method_counts = Counter(quant_methods)
        
        for method, count in method_counts.most_common():
            print(f"  {method}: {count} ({count/has_quant*100:.1f}%)")
        
        # 统计下载量
        total_downloads = sum(r['downloads'] for r in self.results)
        quant_downloads = sum(r['downloads'] for r in self.results if r['has_quant_config'])
        
        print(f"\n下载量统计:")
        print(f"  总下载量: {total_downloads:,}")
        print(f"  有量化配置模型的总下载量: {quant_downloads:,}")
        
    def save_results(self, filename: str = None):
        # 同时保存 JSON 格式
        json_filename = filename
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"JSON 格式已保存到: {json_filename}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='抽样调查 HuggingFace 上量化 LLM 的配置信息'
    )
    parser.add_argument(
        '-n', '--sample-size',
        type=int,
        default=100,
        help='抽样数量 (默认: 100)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=f"quantization_survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help='输出文件名 (默认: quantization_survey_<timestamp>.json)'
    )
    
    args = parser.parse_args()
    
    survey = HFQuantizationSurvey(
        sample_size=args.sample_size
    )
    
    try:
        survey.survey_models()
        survey.analyze_results()
        survey.save_results(args.output)
    except KeyboardInterrupt:
        print("\n\n调查被中断")
        if survey.results:
            print("保存已收集的结果...")
            survey.analyze_results()
            survey.save_results(args.output)


if __name__ == "__main__":
    main()
