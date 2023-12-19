import numpy as np
import pandas as pd
import datetime as dt

PATH_TO_ORIGINAL_DATA = 'data/'
PATH_TO_PROCESSED_DATA = 'data/'

data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'train.tsv', sep='\t', header=None, usecols=[0, 1, 2], dtype={0: np.int32, 1: str, 2: np.int64})
data.columns = ['SessionId', 'ItemId', 'Time']
data['Time'] = data['Time'].apply(lambda x: dt.datetime.utcfromtimestamp(x).timestamp() if isinstance(x, int) else None)

# Filter sessions with length greater than 1
session_lengths = data.groupby('SessionId').size()
data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]

# Filter items with support greater than or equal to 5
item_supports = data.groupby('ItemId').size()
data = data[np.in1d(data.ItemId, item_supports[item_supports >= 5].index)]

# Filter sessions with length greater than or equal to 2
session_lengths = data.groupby('SessionId').size()
data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= 2].index)]

# Split into train and valid sets
tmax = data.Time.max()
session_max_times = data.groupby('SessionId').Time.max()
session_train = session_max_times[session_max_times < tmax - 86400].index
session_valid = session_max_times[session_max_times >= tmax - 86400].index

train = data[np.in1d(data.SessionId, session_train)]
valid = data[np.in1d(data.SessionId, session_valid)]

# Print statistics and save files
print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
train.to_csv(PATH_TO_PROCESSED_DATA + 'train_full.txt', sep='\t', index=False)

print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
train.to_csv(PATH_TO_PROCESSED_DATA + 'train.txt', sep='\t', index=False)

print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
valid.to_csv(PATH_TO_PROCESSED_DATA + 'valid.txt', sep='\t', index=False)

