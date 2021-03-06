import pickle as pickle
import pandas as pd
import torch
import re


# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    # item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
    item = {key: val[idx].clone().detach() for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type, augmented):
  def cleanText(text):
    text = re.sub('[-=+,#/\?:^$.@*\"※~ㆍ!』\\‘|\[\]`\'…》]', '', text)  # %&()<>
    return text

  def entity_marker(s, entity01, entity02):
    entity_code01 = f"<e1> {entity01} </e1>"
    entity_code02 = f'<e2> {entity02} </e2>'

    return s.replace(entity01, entity_code01).replace(entity02, entity_code02)

  if augmented:
    out_dataset = pd.DataFrame({'sentence':dataset[0].apply(cleanText),'entity_01':dataset[1],'entity_02':dataset[2],'label':dataset[3],})
  else:
    label = []
    for i in dataset[8]:
      if i == 'blind':
        label.append(100)
      else:
        label.append(label_type[i])
    
    out_dataset = pd.DataFrame({'sentence':dataset[1].apply(cleanText),'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})

  # out_dataset['sentence'] = out_dataset.apply(lambda x: entity_marker(x['sentence'], x['entity_01'], x['entity_02']), axis=1)
  return out_dataset

# tsv 파일을 불러옵니다.
def load_data(dataset_dir, augmented=False):
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type, augmented)
  
  return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer):
  sep_token = '</s>'
  # sep_token = '[SEP]'
  concat_entity = []
  for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
    temp = ''
    temp = '[CLS]' + e01 + sep_token + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=128,
      add_special_tokens=True,
      )
  return tokenized_sentences