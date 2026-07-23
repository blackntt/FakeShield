import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_json("hf://datasets/zhipeixu/MMTD-Set-34k/MMTD-Set-34k.json")
#print structure of json file at the first row
print(df.head(1).to_dict())
