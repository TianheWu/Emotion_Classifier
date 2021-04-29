from pandas import DataFrame
from __Parameters import *


# save text data
def save_text_data(train_text_pre, train_label):
    texts = [" ".join(words) for words in train_text_pre]
    train_data_save = DataFrame({"text":texts, "label":train_label})
    train_data_save.to_csv(SAVE_DATA_DIR, index=False)
    print("Save text successfully!")
