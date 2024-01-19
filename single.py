import re
import pandas as pd
import json
import statistics
from itertools import chain
import copy
import argparse
from transformers import AutoTokenizer, AutoModel
import pathlib
# import fastllm_pytools
# from fastllm_pytools import llm

# To run this, cmd: python single.py --file xx --checkpoint xx --destination xx

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--destination', type=str, required=True)
parser.add_argument('--start', type=str, required=True)
parser.add_argument('--end', type=str, required=True)

args = parser.parse_args()
FILE = args.file
CKPT = str(pathlib.Path().resolve()) + args.checkpoint
DEST = args.destination
START = int(args.start)
END = int(args.end)

def pred(ds, trait):
    # load testing set
    dataset = pd.read_csv(ds)
    prediction_result = []
    instruction = f"Predict {trait} score in big five personality based on the text. The score is integer ranging from 0 to 10."
    for i in range(len(dataset)):
      row = dataset.iloc[i, :]
      response, history = model.chat(tokenizer, instruction + " : " + row["text"], history=[])
      prediction_result.append(response)

    score_res = []
    for i, label in enumerate(prediction_result):
      number = re.findall("\d+\.\d+", label)
      if len(number) == 0:
        score_res.append(0)
      else:
        score_res.append(float(number[0]))

    return score_res

def grouping(dataset, prediction, trait):
  ds = copy.deepcopy(dataset)
  ds[trait + "_prediction"] = prediction
  prediction_col = trait + "_prediction"
  grouped = ds.groupby(["authorid"]).agg({ prediction_col: lambda x: list(x), 'county_fip': "first"})
  votes = []
  for i, author in enumerate(list(grouped.index)):
    y_cnt, n_cnt = 0, 0
    for val in grouped[prediction_col][author]:
      if val > 5:
        y_cnt += 1
      else:
        n_cnt += 1
    if y_cnt < 0:
        votes.append(-1)
    else:
        votes.append(1)
  grouped[trait + "_2cls"] = votes
  return grouped

tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
model = AutoModel.from_pretrained(CKPT, trust_remote_code=True).half().cuda()
# model = llm.from_hf(model_former, tokenizer, dtype = "float16")
print("model loading completed -----------------------------")

trait = CKPT.split("/")[-2]
for i in range(START, END):
  ds = pd.read_csv(FILE + "batched_sample_" + str(i) + ".csv")
  # res = copy.deepcopy(ds)

  preds = pred(FILE + "batched_sample_" + str(i) + ".csv", trait)
  trait_res = grouping(ds, preds, trait)
  # res = pd.merge(res, trait_res, on="authorid", how="outer")
  trait_res.to_csv(DEST + trait + "_" + str(i) + ".csv")
  print(f"completed inference for {i}-th item -----------------------------")
    
