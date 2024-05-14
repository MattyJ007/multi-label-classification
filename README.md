# Multilabel Classification

## Done
1. Access subset of training data and loop through each row
1. Find largest images portrait and landscape - resize all images to same.
1. Greyscale than normalise (pixel values between 0 - 1)
1. One hot encode labels + check label distribution within training set
1. Preprocess captions - remove stop words + lowercasing + tokenise

## TODO

1. Visualise the data at the different stages
  - Number of labels per image distribution
  - Count caption dataset size (lemma vs stem + together)
1. Explore new models proposed in tute
  - ViT, DeiT, YOLO, CLIP
  - Or ResNet + LSTM?
1. Build models with PyTorch - see if papers have pretrained models available (DEit, ViT, 1.) (maybe check out other multilabel models) (motivation for pretrained model - some of our classes have very few examples - training from scratch would give very poor performance)
   - Use untrained architecture
   - use pretrained architecture
   - (optional) Map pretrained model classes to our classes and check accuracy - play around with whether pretrained model can do multi label
   - pretrained language model
1. Save model to file
1. Add F1 score
  - check which labels have the poorest prediction rate (perhaps add extra training examples for poor predictors)
1. Hyperparameter tuning
1. Ablation tests based on above baselines
- baseline with and without pretraining
- remove captions
- remove images
1. Add improvements
  - Piggyback off of pretrained models
  - cnn layers
  - preprocessing
  - Low rank adaption + parameter efficient fine tuning
  - Split image to increase training sets