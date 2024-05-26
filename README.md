# hate-speech-detection
Deep Learning course project

## Datasets
The base dataset can be downloaded [here](https://hatefulmemeschallenge.com/)

The new dataset can be found [here](https://drive.google.com/drive/folders/196jVnlt4pgWGHH3MqmYWstC_aeAFw7FM?usp=sharing)

## CLIP Model run:
To run the clip model few steps are required so we have created a notebook to help with the task. The notebook is positioned in ./notebooks/ and is called 'setup_clip.ipynb'.
The best model corresponding to CLIP ViT-L/14 (U + A + B) results in the report can be found at this [link](https://drive.google.com/drive/folders/1tvzRkZxXZvHH2XgbZa3U9GhLg_mr7TUE?usp=sharing) alongside the example of code run to generate it.

## Late Fusion Model:

`train.sh` script is responsible for training the late fusion model. Evaluation could be done with the `test.sh` script. If you are running on izar you can use the corresponding `train.run` and `test.run` configuration files.

#### Notes:

- **On clip precomputed embeddings** The augmentation with precomputed embeddings is performed by computing the initial embedding and then 10 different image augmentation and 10 text paraphrases. Then the dataset if a flag is active will sample the original embeddings or a random augmentation with 50% probability.

- **On inference** The goal of our project was to inquire the state of the art on this task and not to create a ready to use model to detect hateful memes. This justify the absence of an interface that allow to load the model and test it.

- **clip library** In the code we also tried to replicate perfectly the results presented in the HateCLIPper presentation paper, to do so we also used a different implementation of clip. However the obtained architecture was not mentioned in the report nor in the poster due to its unimpressive results. In any case to run that part of the code is necessary to `pip install clip`, meanwhile for the other implementation the installation of the clip model is covered in the setup_clip.ipynb notebook.
