# Task 5: MAMI - Multimedia Automatic Misogyny Identification (SemEval 2022)

This repository contains the implementation of a meme classification system using two different transformer-based models: BERT and ViT-BERT, aimed at categorizing memes as either misogynous or non-misogynous. This work was done as part of **SemEval 2022: Task 5 - Sub-task A**.

## Task Description

The task is to classify memes based on their content as either misogynous or non-misogynous. Memes often contain both textual and visual elements, so this task challenges models to handle multi-modal data effectively.

## Repository Structure

- `bert_classifier.ipynb`: This notebook implements a BERT-based classifier. It processes the textual content of memes and classifies them into the two categories.
- `ViTBert.ipynb`: This notebook combines both textual and visual features using a Vision Transformer (ViT) and BERT to improve classification performance by leveraging multimodal inputs.

## Approach

1. **BERT-based Classifier (bert_classifier.ipynb):
- Fine-tuned a pre-trained BERT model for the task of text classification.
- Used the meme's text data to train the model to classify it as misogynous or non-misogynous.
- The model was evaluated using metrics such as accuracy and F1-score.

2. ViT-BERT Model (ViTBert.ipynb):
- Combined textual information from the meme's caption and visual information from the meme's image.
- Used Vision Transformer (ViT) to extract features from the image and BERT for textual features.
- The two features were fused and passed through a classifier for the final prediction.

## Usage

### 1. BERT-based Text Classification:
1. Open `bert_classifier.ipynb`.
2. The notebook will guide you through the steps to fine-tune BERT on the meme text dataset.
3. Run the cells and follow the instructions to train and evaluate the model.

### 2. ViT-BERT Model:
1. Open `ViTBert.ipynb`.
2. This notebook uses a Vision Transformer for image feature extraction and BERT for text feature extraction.
3. Ensure that you have access to the meme image dataset for the model to process both textual and visual inputs.


## Results
The models were evaluated on their ability to classify memes as misogynous or non-misogynous. The BERT-based classifier achieved strong results on text-only data, while the ViT-BERT model demonstrated improved performance by leveraging both text and visual features.

## Acknowledgments
- [SemEval 2022: Task 5 - Sub-task A](https://competitions.codalab.org/competitions/34175) for providing the dataset and task description.
- The authors of [BERT](https://arxiv.org/abs/1810.04805) and [Vision Transformer](https://arxiv.org/abs/2010.11929) for their incredible work on transformer models.
