# event_prediction
Event prediction/Event order prediction

# Prepare data (VTT)
- visual feature (resnet152): http://128.2.210.240:9000/vfeat.tar.gz
- visual feature (resnext): http://128.2.210.240:9000/vfeat_resnext.tar.gz 
- text tokens: vtt_caps.token
- w2v: http://128.2.210.240:9000/w2v.tar.gz

# Run script
```
bash go0.sh
```
or modify the auguments for your experiments. For example
```
model=3head_t_rnn_v_linear_cnn2_dp03_vin_norm
python main.py --is_train --glove --model_name $model --vis_encoder linear --text_encoder rnn --cnn2 --vin_norm
python main.py --load_best --glove --model_name $model --vis_encoder linear --text_encoder rnn --cnn2 --vin_norm
```

# Experiments
- Model check points are under ./model
- Show and compare the stat of different models with parse.py under ./model.


# Performance
|Model|t2i R@1|t2v R@5|t2v R@10|v2t R@1|v2t R@5|v2t R@10| sum R|
|---|---|---|---|---|---|---|---|
|W2VV|6.1|18.7|27.5|11.8|28.9|39.1|132.1|
|SOTA[1]|7.7|22.0|31.8|13.0|30.8|43.3|148.6|
|Ours:3head_t_rnn_v_rnn_tnvn_cnn2_dp03|8.3|24.2|34.2|12.5|32.1|43.6|154.9|
|Ours:3head_t_rnn_v_rnn_tnvn_cnn2_dp03 (next)|11.6|30.7|41.6|18.5|41.5|53.8|197.7|

[1] Dual Encoding for Zero-Example Video Retrieval (CVPR19)


# Findings:
- Visual feature matters
- Complex network not necessary help (eg. 2048-dim embeddings)
- V-RNN and V-Linear similar (temporal not important
- Glove is good enough. W2v in [1] results in similar perofrmance.
- Cannot get BERT work on VTT yet.
