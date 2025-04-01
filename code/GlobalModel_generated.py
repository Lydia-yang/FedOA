import os

import fire
# import gradio as gr
import torch
import transformers
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from audtorch.metrics.functional import pearsonr

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,AutoTokenizer
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

class EvalDataset(Dataset):
    def __init__(self, file, prompter, tokenizer, max_len=512):
        self.prompter = prompter
        self.tokenizer = tokenizer
        with open(file, 'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        line = line.strip()
        ques = json.loads(line)
        sample = ques['instruction']
        prompt = self.prompter.generate_prompt(sample, None)
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # input_ids = inputs["input_ids"][0]
        return prompt, sample

def writeFile(s, path):
    with open(path,'a+',encoding='utf-8') as f1:
        f1.write(s+'\n')

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights_path: str = "",
    lora_config_path: str= "", # provide only the file path, excluding the file name 'adapter_config.json'
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "127.0.0.1",
    share_gradio: bool = False,
    output_file: str="",
    test_file: str="",
    batched: bool = True,
    local: bool = False,
    local_model_path: str = "",
    weight: float = 0.5,
    input_file: str="",
    auto: bool = False,
    half: bool = True,
    max_weight: float = -1,
    instance_num: int = 5,
    emb_type: str="last",
    cal_sim: str="cosine"
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    gpu_count = torch.cuda.device_count()

    if not lora_weights_path.endswith(".bin"):
        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                torch_dtype=torch.float16,
            )
        elif device == "mps":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                device_map={"": device},
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = prepare_model_for_int8_training(model)
        config = LoraConfig.from_pretrained(lora_config_path)
        if gpu_count<3:
            print(gpu_count)
            lora_weights = torch.load(lora_weights_path,map_location=lambda storage, loc: storage.cuda(0))
        else:
            lora_weights = torch.load(lora_weights_path)
        model = PeftModel(model, config)
        
        if local:
            model.add_local_model('local', local_model_path)
            if not auto:
                model.set_local(['local'], [weight,1-weight])
        set_peft_model_state_dict(model,lora_weights,"default")
        model.set_adapter('default')
        del lora_weights

    #exit()

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    tokenizer.padding_side = "left"
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()


    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=80,
        stream_output=True,
        input_ids=None,
        **kwargs,
    ):
        if input_ids is not None:
            input_ids = input_ids.to(device)
            #print(input_ids)
        else:
            prompt = prompter.generate_prompt(instruction, input)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        if len(generation_output.sequences) ==1:
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
            ans = prompter.get_response(output)
        else:
            s = generation_output.sequences.cpu()
            output = tokenizer.batch_decode(s)
            ans = [prompter.get_response(t).split('</s>')[0] for t in output]
        return ans


    def get_weight(sample, lists, num=5, w=0.5, emb_type='last', cal_sim='cosine'):
        model.set_adapter('local')
        print('selected num: ', num)
        samples = random.sample(lists, num)
        sample_list = [samples[i]['instruction'] for i in range(num) ]
        sample_list.append(sample)
        prompt = [prompter.generate_prompt(t, None) for t in sample_list]
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"]
            attention_mask=inputs["attention_mask"]
            # print(input_ids, attention_mask)
            #embs = model.word_embeddings(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True, return_dict=True)
            #print(len(outputs.hidden_states), outputs.hidden_states[0].shape)
            if emb_type == 'avg':
                print("emb type avg")
                emb = torch.mean(outputs.hidden_states[-1], dim=1)
            elif emb_type == 'last':
                print("emb type last")
                emb = outputs.hidden_states[-1][:,-1,:]
            embs = emb.cpu()
        # print(embs.shape)
        cans = embs[-1,:]
        for i in range(num):
            refs = embs[i,:]
            if i==0:
                if cal_sim=='cosine':
                    outs = torch.cosine_similarity(cans,refs,dim=0)
                elif cal_sim=='l2':
                    outs = 1.0 /(1.0+torch.dist(cans,refs,p=2))
                elif cal_sim=='pearsonr':
                    outs = pearsonr(cans,refs).item()
            else:
                if cal_sim=='cosine':
                    outs += torch.cosine_similarity(cans,refs,dim=0)
                elif cal_sim=='l2':
                    outs += 1.0 /(1.0+torch.dist(cans,refs,p=2))
                elif cal_sim=='pearsonr':
                    outs += pearsonr(cans,refs).item()
        outs /= num
        outs *= w
        model.set_adapter('default')
        # print(outs)
        return outs
            

    save = output_file#'out/eval_test.jsonl'
    if batched:
        eval_dataset = EvalDataset(test_file, prompter, tokenizer)
        dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=False)
        #all_text = []
        all_res = []
        for prompts, text in tqdm(dataloader):
            #print(prompts)
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"]
            #print(input_ids.shape)
            res = evaluate(None,input_ids=input_ids)
            #all_text.extend(text)
            all_res.extend(res)
            #print(all_text)
            #print(all_res)
            #break

    if auto:
        lists = []
        tmp_lines = open(input_file).readlines()
        for tmp in tmp_lines:
            tmpd = json.loads(tmp.strip())
            lists.append(tmpd)
        #lists = json.load(open(input_file))
    lines = open(test_file).readlines()
    count=0
    for i,line in enumerate(lines):
        line = line.strip()
        ques = json.loads(line)

        if auto:
            tmpw = 0.5 if half else 1
            if max_weight >0 :
                tmpw = max_weight
            weight = get_weight(ques['instruction'], lists, num=instance_num, w=tmpw, emb_type=emb_type, cal_sim=cal_sim)
            if i==0:
                print("******************", weight,"****************************")
            model.set_local(['local'], [weight,1-weight])
        if not batched:
            res = evaluate(ques['instruction'])
        else:
            res = all_res[i]
        if auto:
            model.unset_local()
        tmp = {}
        tmp['text'] = ques['instruction']
        tmp['answer'] = res
        tmp['category'] = ques['category']
        tmp['task'] = ques['task']
        writeFile(json.dumps(tmp, ensure_ascii=False), save)
        count = count+1
        print('num:', count)
        print("Instruction:", tmp['text'])
        print("Response:", tmp['answer'])
        print("*****************************************************")
        # break



if __name__ == "__main__":
    fire.Fire(main)
