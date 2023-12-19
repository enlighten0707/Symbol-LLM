import os
import json
import tqdm
import torch
import pickle
import numpy as np
from PIL import Image
from threading import Thread

os.environ['CURL_CA_BUNDLE'] = ''

def predict_symbols(hoi_ids, device, B1_dir, A2_dir, B2_dir):
    
    ## load blip
    from utils.lavis.models import load_model_and_preprocess
    device = torch.device("cuda:%d"%device) if torch.cuda.is_available() else "cpu"
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)

    from utils.HICO_utils import hoi_no_inter_all
    for hoi_id in hoi_ids:
        if hoi_id + 1 in hoi_no_inter_all:
            continue
        print("hoi_id: ", hoi_id)
        if not os.path.exists("./DATA/%s/%d"%(B2_dir, hoi_id)):
            os.makedirs("./DATA/%s/%d"%(B2_dir, hoi_id))
        KEY2ENTITY = {}
        key_list = pickle.load(open("./DATA/%s/%d.pkl"%(B1_dir, hoi_id), "rb"))
        for key in tqdm.tqdm(key_list):

            if os.path.exists("./DATA/%s/%d/%s.pkl"%(B2_dir, hoi_id, key)):
                continue
            KEY2ENTITY[key] = []
            raw_image = Image.open("../DATA/hico_20160224_det/test2015/%s"%key).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device) 

            hoi_entities =  json.load(open("./DATA/%s/%d/hoi_entities.json"%(A2_dir, hoi_id), "r"))
            hoi_entities = {v: k for k, v in hoi_entities.items()}
            hoi_entities_list = []
            for i in range(len(hoi_entities)):
                hoi_entities_list.append(hoi_entities[i])
            hoi_entities_list = np.array(hoi_entities_list)
            length = len(hoi_entities_list)
            bz = 12

            for i in range(int(np.ceil(float(length / bz)))):
                cur_entities = hoi_entities_list[bz*i : min(bz*(i+1), length)]
                text_input = []
                text_output = []
                for ent in cur_entities:
                    text_input.append(ent.strip('.') + ". Yes/No?")
                    text_output.append("Yes")
                    text_input.append(ent.strip('.') + ". Yes/No?")
                    text_output.append("No")
                    
                output = model.forward({
                            "image": image,\
                            "text_input": text_input,\
                            "text_output": text_output\
                            })
                loss = output['loss'].detach().cpu().numpy()
                loss = loss.reshape(-1, 2).tolist()
                KEY2ENTITY[key].extend(loss)
            pickle.dump(KEY2ENTITY[key], open("./DATA/%s/%d/%s.pkl"%(B2_dir, hoi_id, key), "wb"))


if __name__ == '__main__':

    hoi_ids = list(range(600))

    ## for clip-trained baseline
    # recog_entity(hoi_ids, device=1, B1_dir="B1_tgt_keys", A2_dir="A2_generate_rule_20230503", B2_dir="B2_recog_entity_20230503")

    ## for blip2 baseline
    # recog_entity(hoi_ids, device=2, B1_dir="B1_tgt_keys_blip2", A2_dir="A2_generate_rule_20230514", B2_dir="B2_recog_entity_blip2_20230514")

    ## for clip baseline
    predict_symbols(hoi_ids, device=6, B1_dir="B1_tgt_keys_clip-zs", A2_dir="A2_generate_rule_20230514", B2_dir="B2_recog_entity_clip-zs_20230514")

    