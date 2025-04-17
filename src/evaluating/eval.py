import argparse
import json
import torch
from utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from training.data import make_supervised_data_module
from training.params import DataArguments

def evaluate_model(model, processor, dataset, device):
    results = []
    model.eval()
    with torch.no_grad():
        for data in dataset:
            inputs = processor(
                text=[data['text']], 
                images=data.get('images', None),
                videos=data.get('videos', None),
                return_tensors="pt"
            ).to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
            
            response = processor.decode(outputs[0], skip_special_tokens=True)
            results.append({
                'input': data,
                'output': response
            })
    
    return results

def main(args):
    device = args.device
    disable_torch_init()
    
    # 加载模型和处理器
    model_name = get_model_name_from_path(args.model_path)
    processor, model = load_pretrained_model(
        model_base=args.model_base,
        model_path=args.model_path,
        device_map=device,
        model_name=model_name
    )

    # 加载自定义数据集
    data_args = DataArguments(
        data_path=args.dataset_path,
        image_folder=args.image_folder
    )
    dataset = make_supervised_data_module(
        model_id=args.model_base,
        processor=processor,
        data_args=data_args
    )['train_dataset']

    # 评估模型
    results = evaluate_model(model, processor, dataset, device)

    # 输出评估结果
    with open(args.output_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--output-path", type=str, default="evaluation_results.json")
    args = parser.parse_args()
    main(args)