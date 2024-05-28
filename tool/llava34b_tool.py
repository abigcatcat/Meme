from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image

class LLava34bTool():
    def __init__(self,model_name_or_path,device_map="auto"):
        
        print(f"Loading model: {model_name_or_path}")
        self.processor = LlavaNextProcessor.from_pretrained(model_name_or_path, use_fast=False)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
            ).eval()
        
    def prepare_input(self,img_path,prompt):
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        prompt = f"<|im_start|>system\nA chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.<|im_end|><|im_start|>user\n<image>\n<{prompt}><|im_end|><|im_start|>assistant\n"


        inputs = self.processor(prompt, image, return_tensors='pt')
        return inputs
    
    def generate(self, img_path,prompt, max_new_tokens, do_sample):
        inputs = self.prepare_input(img_path,prompt).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(**inputs,max_new_tokens=max_new_tokens, do_sample=do_sample)
            response = self.processor.decode(output[0][:], skip_special_tokens=True)
            torch.cuda.empty_cache()
            
        return response