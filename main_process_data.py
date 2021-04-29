from __Data_process import load_data
from __Data_process import text_process
from __Data_process import delete_stop_stem_word
from __Basic_tools import save_text_data
from nltk.corpus import stopwords


train_text, train_label = load_data()
train_text_pre = text_process(train_text)
stop_words = set(stopwords.words("english"))
train_text_pre = delete_stop_stem_word(train_text_pre, stop_words)
save_text_data(train_text_pre, train_label)