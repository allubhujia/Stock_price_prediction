import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

START_DATE = '2013-10-08'
END_DATE = '2018-10-08'
TEST_SIZE = 0.2
RANDOM_STATE = 42

WINDOW_SIZES = [5, 10, 20]

RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2
RF_RANDOM_STATE = 42

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, 'NSE-TATAGLOBAL11.csv')
X_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, 'X_train.npy')
X_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, 'X_test.npy')
Y_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, 'y_train.npy')
Y_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, 'y_test.npy')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pkl')