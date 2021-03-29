import sys
sys.path.insert(0, './datasets')
from Dataset import Dataset




size=[32,128,4]
train_dataset = Dataset(size)
train_dataset.set_info_path()
train_dataset.create_dataset(first_case=1,last_case=5)

val_dataset = Dataset(size)
val_dataset.set_info_path(ds_type="validation")
val_dataset.create_dataset(first_case=1,last_case=1)
