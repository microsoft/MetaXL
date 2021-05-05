## MetaXL: Meta Representation Transformation for Low-resourceCross-lingual Learning

This repo hosts the code for MetaXL, published at NAACL 2021. 
> [MetaXL: Meta Representation Transformation for Low- resource Cross-lingual Learning] (https://arxiv.org/pdf/2104.07908.pdf)
> 
> Mengzhou Xia, Guoqing Zheng, Subhabrata Mukherjee, Milad Shokouhi, Graham Neubig, Ahmed Hassan Awadallah  
> 
> NAACL 2021

MetaXL is a meta-learning framework that learns a main model and a relatively small structure, called representation transformation network (RTN) through a bi-level optimization procedure with the goal to transform representations from auxiliary languages such that it benefits the target task the most.

### Data
Please download [WikiAnn] (https://github.com/afshinrahimi/mmner), [MARC] (https://registry.opendata.aws/amazon-reviews-ml/), [SentiPers] (https://github.com/phosseini/sentipers) and [Sentiraama] (https://ltrc.iiit.ac.in/showfile.php?filename=downloads/sentiraama/) on its corresponding. Please refer to `data/data_index.txt` for data splits. 

### Scripts

The following script shows how to run metaxl on the named entity recognition task on Quechua. 

```bash
python3 mtrain.py \
      --data_dir data_dir \
      --bert_model xlm-roberta-base \
      --tgt_lang qa \
      --task_name panx \
      --train_max_seq_length 200 \
      --max_seq_length 512 \
      --epochs 20 \
      --batch_size 10 \
      --method metaxl \
      --output_dir output_dir \
      --warmup_proportion 0.1 \
      --main_lr 3e-05 \
      --meta_lr 1e-06 \
      --train_size 1000\
      --target_train_size 100 \
      --source_languages en \
      --source_language_strategy specified \
      --layers 12 \
      --struct perceptron \
      --tied  \
      --transfer_component_add_weights \
      --tokenizer_dir None \
      --bert_model_type ori \
      --bottle_size 192 \
      --portion 2 \
      --data_seed 42  \
      --seed 11 \
      --do_train  \
      --do_eval 

```

The following script shows how to run metaxl on the sentiment analysis task on fa. 

``` bash
python3 mtrain.py  \
		--data_dir data_dir \
		--task_name sent \
		--bert_model xlm-roberta-base \
		--tgt_lang fa \
		--train_max_seq_length 256 \
		--max_seq_length 256 \
		--epochs 20 \
		--batch_size 10 \
		--method metaxl \
		--output_dir ${output_dir} \
		--warmup_proportion 0.1 \
		--main_lr 3e-05 \
		--meta_lr 1e-6 \
		--train_size 1000 \
		--target_train_size 100 \
		--source_language_strategy specified  \
		--source_languages en \
		--layers 12 \
		--struct perceptron \
		--tied  \
		--transfer_component_add_weights \
		--tokenizer_dir None  \
		--bert_model_type ori  \
		--bottle_size 192  \
		--portion 2 	\
		--data_seed 42 \
		--seed 11  \
		--do_train  \
		--do_eval
```

### Citation

If you find MetaXL useful, please cite the following paper

```
@inproceedings{xia2021metaxl,
  title={MetaXL: Meta Representation Transformation for Low-resource Cross-lingual Learning},
  author={Mengzhou, Xia and Zheng, Guoqing and Mukherjee, Subhabrata and Shokouhi, Milad and Newbig, Graham and Awadallah, Ahmed Hassan},
  journal={NAACL},
  year={2021},
}
```

This repository is released under MIT License. (See [LICENSE](LICENSE))

	

