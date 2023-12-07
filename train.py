from sentence_transformers import SentenceTransformer, models
from sentence_transformers import InputExample
from util import read_json
from sentence_transformers import losses
from torch.utils.data import DataLoader
from datetime import datetime

output_path = 'training_quora-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 第一步：选择一个已有语言模型
word_embedding_model = models.Transformer('distilroberta-base')
# 第二步：使用一个池化层
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# 将前两步合在一起
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_examples = []
train_data = read_json('data/data.json')
for i, data in enumerate(train_data):
    example = train_data[i]
    train_examples.append(InputExample(texts=[example['vin'], example['car_name']], label=1))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.ContrastiveLoss(model=model)
num_epochs = 10
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=output_path)
