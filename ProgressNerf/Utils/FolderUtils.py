import os
import glob

def last_epoch_from_output_dir(directory:str):
    folders = glob.glob(os.path.join(directory, "epoch_*"))
    epochs = [int(folder.replace(os.path.join(directory, "epoch_"),'')) for folder in folders]
    return max(epochs) if len(epochs) > 0 else -1