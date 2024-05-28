import json
from load_dataset import goat_bench_datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tool.llava13b_tool import LLava13bTool
from PIL import Image
import numpy as np
from harmful import harmful_prompt

# 加载数据集
dataset = goat_bench_datasets('/public/home/jiac/jiac/meme/GOAT-Bench/harmfulness')

# Dataloader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 加载LLava模型
llava = LLava13bTool(model_name_or_path='/public/home/jiac/jiac/agents/llava-hf/llava-v1.6-vicuna-13b-hf')

result_dict = {'id': [], 'gt_label': [], 'pred_label': [], 'output': [], 'error': []}

# 创建一张空白图像
blank_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))

for i, batch in enumerate(dataloader):
    meme_text = batch['text'][0]
    id = batch['id'][0]
    gt_label = int(batch['label'][0])
    try:
        # 生成图片描述
        prompt_des = 'Describe this image in detail.'
        response_des = llava.generate(batch['img_path'], prompt_des, max_new_tokens=100, do_sample=False)
        meme_description = response_des['choices'][0]['text'].strip()

        # 构建提示
        description = f'This is the meme\'s information:\n\n<information>\n{meme_description}\n{meme_text}\n</information>'

        messages = [
            {"role": "system", "content": harmful_prompt['System']},
            {"role": "user", "content": description + harmful_prompt['User']}
        ]

        # 判断表情包是否有害，输入空白图像和描述
        response_judge = llava.generate([blank_image], messages, max_new_tokens=100, do_sample=False)
        judge_text = response_judge['choices'][0]['text'].strip()
        pred_label = 1 if '<answer>YES</answer>' in judge_text else 0

        # 更新结果
        result_dict['id'].append(id)
        result_dict['gt_label'].append(gt_label)
        result_dict['pred_label'].append(pred_label)
        result_dict['output'].append(judge_text)
    except Exception as e:
        result_dict['error'].append((id, str(e)))

# 计算准确率和F1分数
accuracy = accuracy_score(result_dict['gt_label'], result_dict['pred_label'])
f1 = f1_score(result_dict['gt_label'], result_dict['pred_label'], average='macro')

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")

# 保存结果到JSON文件
with open('result_dict.json', 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, ensure_ascii=False, indent=4)
