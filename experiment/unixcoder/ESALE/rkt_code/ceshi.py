import torch
from uniXcoder import UniXcoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("../../../../model/unixcoder-base-nine")
model.to(device)
context = """
# <mask0>
def write_json(data,file_path):
    data = json.dumps(data)
    with open(file_path, 'w') as f:
        f.write(data)
"""
tokens_ids = model.tokenize([context],max_length=512,mode="<encoder-decoder>")
source_ids = torch.tensor(tokens_ids).to(device)
prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=3, max_length=128)
predictions = model.decode(prediction_ids)
print([x.replace("<mask0>","").strip() for x in predictions[0]])