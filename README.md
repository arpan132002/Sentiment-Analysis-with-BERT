# Sentiment Analysis with BERT for Google Play Reviews

## Overview

This project implements sentiment analysis using BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art pre-trained language model developed by Google. The goal is to classify text into sentiment categories such as positive, negative, or neutral. 

## Model Architecture
Here I use pre-trained BERT for binary sentiment analysis on Stanford Sentiment Treebank.

BertEmbeddings: Input embedding layer
BertEncoder: The 12 BERT attention layers
Classifier: Our multi-label classifier with out_features=2, each corresponding to our 2 labels
- BertModel
    - embeddings: BertEmbeddings
      	- word_embeddings: Embedding(28996, 768)
      	- position_embeddings: Embedding(512, 768)
      	- token_type_embeddings: Embedding(2, 768)
      	- LayerNorm: FusedLayerNorm(torch.Size([768])
	- dropout: Dropout = 0.1
    - encoder: BertEncoder
      	- BertLayer
          	- attention: BertAttention
            		- self: BertSelfAttention
              		- query: Linear(in_features=768, out_features=768, bias=True)
              		- key: Linear(in_features=768, out_features=768, bias=True)
               		- value: Linear(in_features=768, out_features=768, bias=True)
              		- dropout: Dropout = 0.1
            	- output: BertSelfOutput(
              		- dense: Linear(in_features=768, out_features=768, bias=True)
              		- LayerNorm: FusedLayerNorm(torch.Size([768]), 
              		- dropout: Dropout =0.1

          	- intermediate: BertIntermediate(
            		- dense): Linear(in_features=768, out_features=3072, bias=True)
          
          	- output: BertOutput
            		- dense: Linear(in_features=3072, out_features=768, bias=True)
            		- LayerNorm: FusedLayerNorm(torch.Size([768])
            		- dropout: Dropout =0.1
 	- pooler: BertPooler
      		- dense: Linear(in_features=768, out_features=768, bias=True)
      		- activation: Tanh()
	- dropout: Dropout =0.1
 	- classifier: Linear(in_features=768, out_features = 2, bias=True)

## Dataset
We'll load the Google Play app reviews dataset

## Features

- **Preprocessing**: Includes tokenization and padding to prepare text data for BERT.
- **BERT Model**: Utilizes BERT for feature extraction and sentiment classification.
- **Fine-Tuning**: Implements fine-tuning of the BERT model on the sentiment analysis dataset.
- **Evaluation**: Provides evaluation metrics including accuracy, precision, recall, and F1-score.

## Acknowledgements
- **BERT Model**
- **Hugging Face Transformers**
