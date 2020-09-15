from ImageStyleTransferer import ImageStyleTransferer
import os
import json
import torch

# strings

# config

PARAMETERS_EXAMPLE = '''{
  "LayerParameters": [ 1.0, 0.75, 0.5, 0.9, 0.5 ],
  "MaxImageSize": 400,
  "IterationNumber": 5000,
  "ShowEvery": 500,
  "LearningRate": 0.03,
  "BetaParameter": 1000000
}
'''

PARAMETERS_FILE_NAME = "parameters.json"

CONFIG_NOT_FOUND_MSG = f"{PARAMETERS_FILE_NAME} not found! Create a {PARAMETERS_FILE_NAME} and put the parameters into it.\nExample : {PARAMETERS_EXAMPLE}"

# model initialization

VGG_INITIALIZING_MSG = "Initializing VGG"

VGG_INITIALIZED_MSG = "VGG initialized"

# device

CUDA = "cuda"

LOADING_VGG_TO_GPU_MSG = "Loading model to GPU"

LOADED_VGG_TO_GPU_MSG = "Model loaded to GPU"

CUDA_NOT_SUPPERTED_MSG = f"{CUDA} is not supported by your machine. All calculations will be done on the CPU. It may take a little longer."

def Main():

    if not os.path.exists(PARAMETERS_FILE_NAME):
        print(CONFIG_NOT_FOUND_MSG)
        return

    with open(PARAMETERS_FILE_NAME, "r") as read_file:
        data = json.load(read_file)

    contentImagePath = "Samples\\cute-animals-4k-kitten-wallpaper-preview.jpg"

    styleImagePath = ["amples\\style_tron.png","Samples\\style_van_gog_zvezdnaya_noch.png","Samples\\style_colored_frac.png"]

    print(VGG_INITIALIZING_MSG)

    ist = ImageStyleTransferer(**data)

    print(VGG_INITIALIZED_MSG)

    if torch.cuda.is_available():

        print(LOADING_VGG_TO_GPU_MSG)

        ist.SetDevice(torch.device(CUDA))

        print(LOADED_VGG_TO_GPU_MSG)

    else:

        print(CUDA_NOT_SUPPERTED_MSG)

    for stylePath in styleImagePath:
        ist.Process(contentImagePath,stylePath)


if __name__ == "__main__":
    Main()