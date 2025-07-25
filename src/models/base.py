# import librosa
# import os
# import typing

# import logging

# logger = logging.getLogger(__name__)


# class BaseAugmentor:
#     """
#     Basic augmentor class requires these config:
#     aug_type: str, augmentation type
#     output_path: str, output path
#     out_format: str, output format
#     """

#     def __init__(self, config: dict):
#         """
#         This method initialize the `BaseAugmentor` object.
#         """
#         self.config = config
#         self.aug_type = config["aug_type"]
        
#         self.output_path = config["output_path"]
#         self.out_format = config["out_format"]
#         self.augmented_audio = None
#         self.data = None
#         self.sr = 16000

#     def load(self, input_path: str):
#         """
#         Load audio file and normalize the data
#         Librosa done this part
#         self.data: audio data in numpy array (librosa load)
#         :param input_path: path to the input audio file      
#         """
#         self.input_path = input_path
#         self.file_name = self.input_path.split("/")[-1].split(".")[0]
#         # load with librosa and auto resample to 16kHz
#         self.data, self.sr = librosa.load(self.input_path, sr=self.sr)

#         # Convert to mono channel
#         self.data = librosa.to_mono(self.data)

#     def transform(self):
#         """
#         Transform audio data (librosa load) to augmented audio data (pydub audio segment)
#         Note that self.augmented_audio is pydub audio segment
#         """
#         raise NotImplementedError

#     def save(self):
#         """
#         Save augmented audio data (pydub audio segment) to file
#         self.out_format: output format
#         This done the codec transform by pydub
#         """
#         self.augmented_audio.export(
#             os.path.join(self.output_path, self.file_name + "." + self.out_format),
#             format=self.out_format,
#         )



import librosa
import os
import typing
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
import torch

logger = logging.getLogger(__name__)


class BaseAugmentor(Dataset):
    """
    Basic augmentor class requires these config:
    aug_type: str, augmentation type
    output_path: str, output path
    out_format: str, output format
    """

    def __init__(self, config: dict, input_paths, file_names):
        """
        This method initialize the `BaseAugmentor` object.
        """
        self.config = config
        self.aug_type = config["aug_type"]
        
        self.output_path = config["output_path"]
        self.out_format = config["out_format"]
        self.adv_audio = None
        # self.rm_audio = []
        self.rm_audio = None
        self.data = None
        self.sr = 16000
        self.attack_type1 = config["adv_method1"]
        self.input_paths = input_paths
        self.file_names = file_names
        self.data_batch = []
        self.rm_audio_batch = []
        self.adv_audio_batch = []
        self.batch_size = config["batch_size"]
        self.batched_filenames = []

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        path = self.input_paths[idx]
        data, _ = librosa.load(path, sr=self.sr)
        data = librosa.to_mono(data)
        return data
 
    def collate_fn(self, batch):
        """
        Function to collate the batch of data.
        This function pads the input audio data to ensure equal length.
        """
        # Find the maximum length in the batch
        max_len = max([len(data) for data in batch])
        # Pad each data sample to the maximum length
        padded_batch = [np.pad(data, (0, max_len - len(data))) for data in batch]
        # Convert the padded_batch to tensor
        return np.array(padded_batch)
    
    def load(self, input_path: str):
        """
        Load audio file and normalize the data
        Librosa done this part
        self.data: audio data in numpy array (librosa load)
        :param input_path: path to the input audio file      
        """
        self.input_path = input_path
        self.file_name = self.input_path.split("/")[-1].split(".")[0]
        # load with librosa and auto resample to 16kHz
        self.data, self.sr = librosa.load(self.input_path, sr=self.sr)

        # Convert to mono channel
        self.data = librosa.to_mono(self.data)


    def load_batch(self, input_paths: typing.List[str]):
        """
        Load audio files and normalize the data
        Librosa done this part
        self.data: audio data in numpy array (librosa load)
        :param input_paths: list of paths to the input audio files
        """
        self.file_names = [path.split("/")[-1].split(".")[0] for path in input_paths]
        dataset = BaseAugmentor(self.config, input_paths, self.file_names)
        # data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, collate_fn=self.collate_fn)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        self.data_batch = data_loader

        self.batched_filenames = [self.file_names[i:i+self.batch_size] for i in range(0, len(self.file_names), self.batch_size)]

        # print("batched_filenames: ", batched_filenames)
        # for batch in batched_filenames:
        #     print(batch)

        # for i, batch in enumerate(self.data_batch):
        #     print(i)
        #     print(batch)


    def transform(self):
        """
        Transform audio data (librosa load) to augmented audio data (pydub audio segment)
        Note that self.augmented_audio is pydub audio segment
        """
        raise NotImplementedError

    def save(self):
        """
        Save augmented audio data (pydub audio segment) to file
        self.out_format: output format
        This done the codec transform by pydub
        """

        if self.adv_audio is not None:
            self.adv_audio.export(
                os.path.join(self.output_path, self.file_name + "_" + self.attack_type1 + "." + self.out_format),
                format=self.out_format,
            )

        if self.rm_audio is not None:
            self.rm_audio.export(
                os.path.join(self.output_path, self.file_name + "_" + self.attack_type1 + "_rm." + self.out_format),
                format=self.out_format,
            )


    def save_batch(self, i_batch):
        """
        Save augmented audio data (pydub audio segment) to file
        self.out_format: output format
        This done the codec transform by pydub
        """

        for filename, adv_audio, rm_audio in zip(self.batched_filenames[i_batch], self.adv_audio_batch, self.rm_audio_batch):
            print("filename: ", filename)
            print("adv_audio: ", adv_audio)
            print("rm_audio: ", rm_audio)
            if adv_audio is not None:
                adv_audio.export(
                    os.path.join(self.output_path, filename + "_" + self.attack_type1 + "." + self.out_format),
                    format=self.out_format,
                )
            if rm_audio is not None:
                rm_audio.export(
                    os.path.join(self.output_path, filename + "_" + self.attack_type1 + "_rm." + self.out_format),
                    format=self.out_format,
                )



#############################################################################################################################################
# attack with 2 type

# class BaseAugmentor(Dataset):
#     """
#     Basic augmentor class requires these config:
#     aug_type: str, augmentation type
#     output_path: str, output path
#     out_format: str, output format
#     """

#     def __init__(self, config: dict, input_paths, file_names):
#         """
#         This method initialize the `BaseAugmentor` object.
#         """
#         self.config = config
#         self.aug_type = config["aug_type"]
        
#         self.output_path = config["output_path"]
#         self.out_format = config["out_format"]
#         self.adv_audio = None
#         # self.rm_audio = []
#         self.rm_audio = None
#         self.data = None
#         self.sr = 16000
#         self.attack_type1 = config["adv_method1"]
#         self.attack_type2 = config["adv_method2"]
#         self.input_paths = input_paths
#         self.file_names = file_names
#         self.data_batch = []
#         self.rm_audio_batch = []
#         self.adv_audio_batch = []
#         self.batch_size = config["batch_size"]
#         self.batched_filenames = []

#     def __len__(self):
#         return len(self.input_paths)

#     def __getitem__(self, idx):
#         path = self.input_paths[idx]
#         data, _ = librosa.load(path, sr=self.sr)
#         data = librosa.to_mono(data)
#         return data
 
#     def collate_fn(self, batch):
#         """
#         Function to collate the batch of data.
#         This function pads the input audio data to ensure equal length.
#         """
#         # Find the maximum length in the batch
#         max_len = max([len(data) for data in batch])
#         # Pad each data sample to the maximum length
#         padded_batch = [np.pad(data, (0, max_len - len(data))) for data in batch]
#         # Convert the padded_batch to tensor
#         return np.array(padded_batch)
    
#     def load(self, input_path: str):
#         """
#         Load audio file and normalize the data
#         Librosa done this part
#         self.data: audio data in numpy array (librosa load)
#         :param input_path: path to the input audio file      
#         """
#         self.input_path = input_path
#         self.file_name = self.input_path.split("/")[-1].split(".")[0]
#         # load with librosa and auto resample to 16kHz
#         self.data, self.sr = librosa.load(self.input_path, sr=self.sr)

#         # Convert to mono channel
#         self.data = librosa.to_mono(self.data)


#     def load_batch(self, input_paths: typing.List[str]):
#         """
#         Load audio files and normalize the data
#         Librosa done this part
#         self.data: audio data in numpy array (librosa load)
#         :param input_paths: list of paths to the input audio files
#         """
#         self.file_names = [path.split("/")[-1].split(".")[0] for path in input_paths]
#         dataset = BaseAugmentor(self.config, input_paths, self.file_names)
#         # data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, collate_fn=self.collate_fn)
#         data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
#         self.data_batch = data_loader

#         self.batched_filenames = [self.file_names[i:i+self.batch_size] for i in range(0, len(self.file_names), self.batch_size)]

#         # print("batched_filenames: ", batched_filenames)
#         # for batch in batched_filenames:
#         #     print(batch)

#         # for i, batch in enumerate(self.data_batch):
#         #     print(i)
#         #     print(batch)


#     def transform(self):
#         """
#         Transform audio data (librosa load) to augmented audio data (pydub audio segment)
#         Note that self.augmented_audio is pydub audio segment
#         """
#         raise NotImplementedError

#     def save(self):
#         """
#         Save augmented audio data (pydub audio segment) to file
#         self.out_format: output format
#         This done the codec transform by pydub
#         """

#         if self.adv_audio is not None:
#             self.adv_audio.export(
#                 os.path.join(self.output_path, self.file_name + "_" + self.attack_type1 + "." + self.out_format),
#                 format=self.out_format,
#             )

#         if self.rm_audio is not None:
#             self.rm_audio.export(
#                 os.path.join(self.output_path, self.file_name + "_" + self.attack_type1 + "_rm." + self.out_format),
#                 format=self.out_format,
#             )


#     def save_batch(self, i_batch):
#         """
#         Save augmented audio data (pydub audio segment) to file
#         self.out_format: output format
#         This done the codec transform by pydub
#         """

#         for filename, adv_audio, rm_audio in zip(self.batched_filenames[i_batch], self.adv_audio_batch, self.rm_audio_batch):
#             print("filename: ", filename)
#             print("adv_audio: ", adv_audio)
#             print("rm_audio: ", rm_audio)
#             if adv_audio is not None:
#                 adv_audio.export(
#                     os.path.join(self.output_path, filename + "_" + self.attack_type1 + "." + self.out_format),
#                     format=self.out_format,
#                 )
#             if rm_audio is not None:
#                 rm_audio.export(
#                     os.path.join(self.output_path, filename + "_" + self.attack_type1 + "_rm." + self.out_format),
#                     format=self.out_format,
#                 )
