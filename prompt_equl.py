import json

my_label=[]
my_prompt = []
with open('experiment/ds-coder-sentences/python_nl_tryCopyDeepTemplate.json','r') as my_in:
    datas = json.load(my_in)
    for data in datas:
        my_label.append(data['label'])
        my_prompt.append(data['prompt'])

llama_label=[]
llama_prompt = []
with open('finetune/eval_result/PYTHON_sentences/generated_predictions.jsonl','r') as my_in:
    for line in my_in:
        data = json.loads(line)
        llama_label.append(data['label'])
        llama_prompt.append(data['prompt'])

prompt_num = 0
for i in range(len(my_prompt)):
    if my_prompt[i] not in llama_prompt:
        prompt_num += 1
        print("faith id",str(i)+'\n'+my_prompt[i])

label_num = 0
for i in range(len(my_label)):
    if my_label[i] not in llama_label:
        label_num += 1
        print("faith id",str(i)+'\n'+my_label[i])

print("********************")

print(prompt_num)
print(label_num)