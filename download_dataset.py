from dotenv import load_dotenv
import os
import tensorflow_datasets as tfds

# Load environment variables from .env file
load_dotenv()

# Now you can access the TFDS_DATA_DIR variable
tfds_data_dir = os.getenv("TFDS_DATA_DIR")
tfds_dataset_name = os.getenv("DATASET")
print(
    f"The dataset {tfds_dataset_name} will be downloaded to the directory {os.path.join(os.getcwd(), tfds_data_dir, tfds_dataset_name)}")
tfds.load('div2k/bicubic_x4', with_info=True)
print("DONE!")
print(f"The dataset {tfds_dataset_name} downloaded at ->  {os.path.join(os.getcwd(), tfds_data_dir, tfds_dataset_name)}")
