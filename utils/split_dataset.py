import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to original dataset
yes_dir = '/home/senghuyjr11/Projects/brain_tumor_dataset/yes'
no_dir = '/home/senghuyjr11/Projects/brain_tumor_dataset/no'

# Destination directories for train, val, and test sets
train_dir = '../dataset/train'
val_dir = '../dataset/val'
test_dir = '../dataset/test'

# Create directories for train, validation, and test
for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(folder, 'yes'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'no'), exist_ok=True)


# Function to split and copy files
def split_and_copy_data(source_dir, category, train_dir, val_dir, test_dir, test_size=0.2, val_size=0.1):
    files = os.listdir(source_dir)
    train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=val_size, random_state=42)

    # Copy files to train, val, and test directories
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, category, file))
    for file in val_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, category, file))
    for file in test_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, category, file))


# Split and copy data for 'yes' and 'no' categories
split_and_copy_data(yes_dir, 'yes', train_dir, val_dir, test_dir)
split_and_copy_data(no_dir, 'no', train_dir, val_dir, test_dir)
