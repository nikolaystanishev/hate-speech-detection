# hate-speach-detection
Deep Learning course project

## Datasets
The base dataset can be downloaded [here](https://hatefulmemeschallenge.com/)
The new dataset can be found [here](https://drive.google.com/drive/folders/196jVnlt4pgWGHH3MqmYWstC_aeAFw7FM?usp=sharing)

## CLIP Model run:
To run the clip model few steps are required so we have created a notebook to help with the task. The notebook is positioned in ./notebooks/ and is called 'setup_clip.ipynb'

#### Notes:

- **On clip precomputed embeddings** The augmentation with precomputed embeddings is performed by computing the initial embedding and then 10 different image augmentation and 10 text paraphrases. Then the dataset if a flag is active will sample the original embeddings or a random augmentation with 50% probability.

- **On inference** The goal of our project was to inquire the state of the art on this task and not to create a ready to use model to detect hateful memes. This justify the absence of an interface that allow to load the model and test it.
