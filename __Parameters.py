SAVE_DATA_DIR = "/home/ipsc/data-pool/wutianhe/Emotion/imdb_data.csv"
SAVE_MODEL_PARAMETERS = "/home/ipsc/data-pool/wutianhe/Emotion/model.pkl"
GLOVE = "glove.6B.100d"

BATCH_SIZE = 32
EMBEDDING_DIM = 100 # 词向量维度
N_FILITERS = 100 # 每个卷积核的个数
FILTER_SIZES = [3, 4, 5] # 卷积核的大小
OUTPUT_DIM = 1 # 输出维度
DROPOUT = 0.5
EPOCHS = 20