# Finetune HuggingFace's T5 

This repository allows you to finetune HuggingFace's T5 implementation on Neural Machine Translation.

The first thing to do is to specify configurations in a config file. Therem you will input desired pretrained model size, training details, data paths, model prefix, and so on. Check out t5/configs for example configuration files. 

Assuming your desired experiment name is en_pd and config file is in t5_config.yml, run the finetune_t5.py file as follows:
```
python t5/finetune_t5.py --experiment_name=en_pd --config_path=t5_config.yml
```

This command begins finetuning T5 on your input parallel data and saves the experiment outputs to a created directory of experiment_name + current date and time : 
```
$PWD/experiments/en_pd_{date and time}
```

After training, you can evaluate an input test set (assuming src.txt and tgt.txt) with the following command:
```
python evaluate_test.py --experiment_path=experimets/en_pd_{date and time} --src_test=src.txt --tgt_test=tgt.txt
```

Make sure to run the help command below to see a full description and format of all input flags
```
python evaluate_test.py --helpshort
```
Other flags:
- --save_as_pretrained : boolean - If True, save the loaded model as a huggingface pretrained model
- --use_cuda : boolean - If True, will use GPU (if available). 
