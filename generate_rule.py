import os
import re
import time
import glob
import tqdm
import torch
import requests
import pickle
import numpy as np
import json
import jsonlines
from PIL import Image
import queue
import clip

device = "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def call_chatgpt(prompt):       
    
    content = [
        {"role": "system", "content": "You are helping me understanding human activities in a picture."},
        {"role": "user", "content": prompt},
    ]
                              
    headers = {
        'content-type': 'application/json',
        "Authorization": "Bearer [openai_api_key]",
    }
    url = 'https://api.openai-proxy.com/v1/chat/completions'
    data = {"model": "gpt-3.5-turbo", "messages": content,"temperature": 0.7}
    
    while True:
        response = requests.post(url, headers=headers, json=data)
        try:
            response = json.loads(response.content)["choices"][0]["message"]
            ans = response['content']
            return ans
        except:
            continue

def get_initial_symbols(act, act_ing, obj, save_dir):
    
    save_dir_txt = os.path.join(save_dir, "prompt_initial_symbols.txt")
    if os.path.exists(save_dir_txt):
        with open(save_dir_txt, "r") as f:
            ans = f.read()
    else:
        question = """In a picture, a person is %s. What are his hands doing? 
Answer with 5 concise, highly-related pharses. The format is "<hands> <verb> <object>".
Output Format: 
1. xxx
2. xxx
3. xxx
4. xxx
5. xxx"""%act_ing
        
        ans = call_chatgpt(question)
        with open(save_dir_txt, "w") as f:
            f.write(ans)
            
    initial_symbols = []
    for line in ans.split('\n'):
        for idx in range(1, 6):
            line = line.strip('%d.'%idx)
        line = line.strip()
        if len(line) > 0:
            initial_symbols.append(line)
    return initial_symbols


def entailment_check(act, act_ing, obj, save_dir, symbol_idx, premise_symbols):
    
    premise_symbols = ["The person is " + sym for sym in premise_symbols]
    premise = ". ".join(premise_symbols)
    if obj is None:
        question = """In a picture, there is %s. %s. Estimate the probability that he is %s at the same time.
Choose from: (a) 0.1, (b) 0.5, (c) 0.7, (d) 0.9, (e) 0.95, (f) unknown. 
Output Format: a/b/c/d/e/f.
"""%(obj, premise, act_ing)
    else:
        question = """In a picture, %s. Estimate the probability that he is %s at the same time.
Choose from: (a) 0.1, (b) 0.5, (c) 0.7, (d) 0.9, (e) 0.95, (f) unknown. 
Output Format: a/b/c/d/e/f.
"""%(premise, act_ing) 
    
    probs = []
    scores = [0.1, 0.5, 0.7, 0.9, 0.95, 0]
    while len(probs) < 3:
        ans = call_chatgpt(question)
        for i, option in enumerate(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]):
            if ans.find(option) > -1:
                probs.append(scores[i])
    probs = np.array(probs)
    return probs.mean()
        

def rule_extention(act, act_ing, obj, save_dir, symbol_idx, premise_symbols):
    
    premise_symbols = ["[The person is " + sym + "] " for sym in premise_symbols]
    premise = "AND ".join(premise_symbols)
    if obj is None:
        question = """In a picture, IF %s AND [condition] THEN [The person is %s.]
[condition] is one concise pharses. The format is "<The person's hands/arms/hip/legs/feet> <verb> <object>". What is [condition]?
Output Format: 
[condition] is: [xxx].
"""%(premise, act_ing)
    else:
        question = """In a picture, there is %s. IF %s AND [condition] THEN [The person is %s.]
[condition] is one concise pharses. The format is "<The person's hands/arms/hip/legs/feet> <verb> <object>". What is [condition]?
Output Format: 
[condition] is: [xxx].
"""%(obj, premise, act_ing)

    ans = call_chatgpt(question)
    new_symbol = ans.strip("[condition] is: ")
    new_symbol = new_symbol.strip('\"').strip()
    return new_symbol


def compare_semantic_similarity(text_list_1, text_list_2):
    
    def get_clip_text_embed(text_list):
        text = clip.tokenize(text_list, truncate=True).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text)
        return text_features
    
    text_embed_1 = get_clip_text_embed(text_list_1)
    text_embed_2 = get_clip_text_embed(text_list_2)
    sim = (text_embed_1 / text_embed_1.norm(dim=1, keepdim=True))@\
        (text_embed_2 / text_embed_2.norm(dim=1, keepdim=True)).t()
    sim = sim.numpy()
    return sim
    
def get_symbols_and_rules(act, act_ing, obj, save_root_dir):
    
    symbols_list = []
    rules = []
    rules_entailment = []
    
    save_dir = os.path.join(save_root_dir, "%s"%act)
    if os.path.exists(os.path.join(save_dir, "hoi_entities.json")) and os.path.exists(os.path.join(save_dir, "hoi_rule.json")):
        return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    initial_symbols = get_initial_symbols(act, act_ing, obj, save_dir)
    
    cand_symbols = queue.Queue()
    for sym in initial_symbols:
        symbols_list.append(sym)
        cand_symbols.put(sym)
    
    loop_cnt = 0
    while not cand_symbols.empty() and loop_cnt < 15:
        
        loop_cnt += 1
        sym = cand_symbols.get()
        premise_symbols = []
        premise_symbols.append(sym)
        symbol_idx = np.where(np.array(symbols_list) == sym)[0][0]

        entailment_score = entailment_check(act, act_ing, obj, save_dir, symbol_idx, premise_symbols)
        new_symbol = rule_extention(act, act_ing, obj, save_dir, symbol_idx, premise_symbols)
        premise_symbols.append(new_symbol)

        rule_entailment = []
        rule_entailment.append(None)
        rule_entailment.append(entailment_score)

        while len(premise_symbols) < 5:
            entailment_score = entailment_check(act, act_ing, obj, save_dir, symbol_idx, premise_symbols)
            rule_entailment.append(entailment_score)
            if entailment_score > 0.9:
                break
            new_symbol = rule_extention(act, act_ing, obj, save_dir, symbol_idx, premise_symbols)
            premise_symbols.append(new_symbol)

        similarity = compare_semantic_similarity(premise_symbols, symbols_list)

        rule = []
        rule.append(1)
        for idx, sym in enumerate(premise_symbols):
            if similarity[idx].max() < 0.95:
                symbols_list.append(sym)
                cand_symbols.put(sym)
                rule.append(sym)
            else:
                rule.append(symbols_list[np.argmax(similarity[idx])])
        rules.append(rule)
        rules_entailment.append(rule_entailment)
    
        json.dump(symbols_list, open(os.path.join(save_dir, "hoi_entities_ori.json"), "w"), indent=4)
        json.dump(rules, open(os.path.join(save_dir, "hoi_rule_ori.json"), "w"), indent=4)
        json.dump(rules_entailment, open(os.path.join(save_dir, "rule_entailment_ori.json"), "w"), indent=4)

def post_process(act, act_ing, obj, save_root_dir):
    
    save_dir = os.path.join(save_root_dir, "%s"%act)
    if os.path.exists(os.path.join(save_dir, "hoi_entities.json")) and os.path.exists(os.path.join(save_dir, "hoi_rule.json")):
        return
    symbols_ori = json.load(open(os.path.join(save_dir, "hoi_entities_ori.json"), "r"))
    rules_ori = json.load(open(os.path.join(save_dir, "hoi_rule_ori.json"), "r"))
    
    symbol_dict = {}
    symbols = np.unique(np.array(symbols_ori))
    for i, symbol in enumerate(symbols):
        symbol_dict[symbol] = i
    
    rules = []
    for rule in rules_ori:
        rule = np.unique(np.array(rule))
        for r in rule[1:]:
            assert r in symbol_dict.keys()
        rules.append(rule.tolist())

    json.dump(symbol_dict, open(os.path.join(save_dir, "hoi_entities.json"), "w"), indent=4)
    json.dump(rules, open(os.path.join(save_dir, "hoi_rule.json"), "w"), indent=4)
    
    
if __name__ == '__main__':
    
    act = "push a bicycle"
    act_ing = "pushing an airplane"
    obj = "a bicycle"
    save_root_dir = "rules/HICO"
    
    get_symbols_and_rules(act, act_ing, obj, save_root_dir)
    post_process(act, act_ing, obj)