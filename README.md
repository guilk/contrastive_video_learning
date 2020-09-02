## Contrastive Representation Learning for YT Structural Videos

In this repo, we propose a framework to learn video representations from large collections of unlabeled YT structural videos by using contrastive learning.

We train our model on [HowTo100M](https://arxiv.org/abs/1906.03327) dataset and evaluate our model on [YouCookII](http://youcook2.eecs.umich.edu/) dataset .

We evaluate our learned representations on two instructional video understanding tasks. 
1) We use verb/noun classification task to evaluate the quality of our learned representations.
2) We use text-to-video retrieval task to evaluate the usefulness of our learned representations on downstream tasks. 
### Data Pre-processing
To download HowTo100M dataset and YouCookII dataset, run
```
python code/download_howto100m.py
```
and 
```
python code/download_youcookii_videos.py
```
### Video representation extraction
After we download all the videos, we can extract two pre-trained representations [ResNet152](https://arxiv.org/abs/1512.03385) and [3D ResNet-18](https://arxiv.org/pdf/1708.07632.pdf).
First of all we need to generate a csv containing the list of videos we
want to process. For instance, if we have video1.mp4 and video2.webm to process,
we will need to generate a csv of this form:

```sh
video_path,feature_path
video1.mp4, path_of_video1_features.npy
video2.webm, path_of_video2_features.npy
```

And then just simply run:

```sh
python extract.py --csv=input.csv --type=2d --batch_size=64 --num_decoding_thread=4
```
This command will extract 2d video feature for video1.mp4 (resp. video2.webm) at path_of_video1_features.npy (resp. path_of_video2_features.npy) in
a form of a numpy array.
To get feature from the 3d model instead, just change type argument 2d per 3d.
The parameter --num_decoding_thread will set how many parallel cpu thread are used for the decoding of the videos.

### Contrastive Representation Learning
To do unsupervised pre-training of a 3D ResNet-18 model on HowTo100M dataset in an 8-gpu machine, we run
```
python main_moco.py \
  -a resnet18 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your howto100m-folder with train and val folders]
```
### Verb/noun Classification
We collect action and object labels, derived from the ground truth captions of YouCookII dataset. 
We run an off-the-shelf part-of-speech tagger on the ground truth captions to retrieve the 100 most common nouns and 45 most common verbs, and use these to derive ground truth labels. 
```
python code/create_labels_youcook.py
```
The verbs and nouns of interest can be found in code/youcook2_verbs.txt and code/youcook2_objects.text. After generating the labels, we run 
```
python baseline_classifiers/linear_baseline.py
```
to compare the quality of different kinds of representations. We use top-1 and top-5 accuracy as our evaluation metrics.

### Text-to-video Retrieval
We use the clip-caption pairs from YouCookII dataset to train our text2video retrieval model. The command line that we use to train our model is 

```
python text2video_retrieval/main.py --is_train --glove --model_name $model --vis_encoder linear --text_encoder rnn --cnn2 --vin_norm
```
To evaluate our retrieval model with different types of video representations, we run
```
python text2video_retrieval/main.py --load_best --glove --model_name $model --vis_encoder linear --text_encoder rnn --cnn2 --vin_norm
```
We use recall@k to compare the results.


