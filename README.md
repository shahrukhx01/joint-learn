<!--- BADGES: START --->
[![GitHub - License](https://img.shields.io/github/license/shahrukhx01/joint-learn?logo=github&style=flat&color=green)][#github-license]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sentence-transformers?logo=pypi&style=flat&color=blue)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/sentence-transformers?logo=pypi&style=flat&color=orange)][#pypi-package]

[#github-license]: https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE
[#pypi-package]: https://pypi.org/project/sentence-transformers/
[#conda-forge-package]: https://anaconda.org/conda-forge/sentence-transformers
[#docs-package]: https://www.sbert.net/
<!--- BADGES: END --->

# Joint Learn: A python toolkit for task-specific weight sharing for sequence classification

This framework provides an easy method to compute dense vector representations for **sentences**, **paragraphs**, and **images**. The models are based on transformer networks like BERT / RoBERTa / XLM-RoBERTa etc. and achieve state-of-the-art performance in various task. Text is embedding in vector space such that similar text is close and can efficiently be found using cosine similarity.

We provide an increasing number of **[state-of-the-art pretrained models](https://www.sbert.net/docs/pretrained_models.html)** for more than 100 languages, fine-tuned for various use-cases.

Further, this framework allows an easy  **[fine-tuning of custom embeddings models](https://www.sbert.net/docs/training/overview.html)**, to achieve maximal performance on your specific task.


## Usage

### LSTM
[Full Example](https://github.com/shahrukhx01/joint-learn/blob/main/joint_learn/examples/hindi_bengali_hasoc_2019/joint_learn_lstm.py)
```python
    ## init jl lstm
    jl_lstm = JLLSTMClassifier(
        batch_size=batch_size,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        embedding_size=embedding_size,
        dataset_hyperparams=dataset_hyperparams,
        device=device,
    )

    ## define optimizer and loss function
    optimizer = torch.optim.Adam(params=jl_lstm.parameters())

    train_model(
        model=jl_lstm,
        optimizer=optimizer,
        dataloaders=jl_dataloaders,
        max_epochs=max_epochs,
        config_dict={"device": device, "model_name": "jl_lstm"},
    )
```

### LSTM Transformer Encoder
[Full Example](https://github.com/shahrukhx01/joint-learn/blob/main/joint_learn/examples/hindi_bengali_hasoc_2019/joint_learn_lstm_transformer.py)
```python
    ## init jl transformer lstm
    jl_lstm = JLLSTMTransformerClassifier(
        batch_size=batch_size,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        embedding_size=embedding_size,
        nhead=nhead,
        transformer_hidden_size=transformer_hidden_size,
        transformer_layers=transformer_layers,
        dataset_hyperparams=dataset_hyperparams,
        device=device,
        max_seq_length=max_seq_length,
    )

    ## define optimizer and loss function
    optimizer = torch.optim.Adam(params=jl_lstm.parameters())

    train_model(
        model=jl_lstm,
        optimizer=optimizer,
        dataloaders=jl_dataloaders,
        max_epochs=max_epochs,
        config_dict={"device": device, "model_name": "jl_lstm"},
    )
```

### LSTM Self-Attention
[Full Example](https://github.com/shahrukhx01/joint-learn/blob/main/joint_learn/examples/hindi_bengali_hasoc_2019/joint_learn_lstm_attention.py)
```python
    ## init jl lstm self-attention
    jl_lstm = JLLSTMAttentionClassifier(
        batch_size=batch_size,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        embedding_size=embedding_size,
        dataset_hyperparams=dataset_hyperparams,
        bidirectional=bidirectional,
        fc_hidden_size=fc_hidden_size,
        self_attention_config=self_attention_config,
        device=device,
    )

    ## define optimizer and loss function
    optimizer = torch.optim.Adam(params=jl_lstm.parameters())

    train_model(
        model=jl_lstm,
        optimizer=optimizer,
        dataloaders=jl_dataloaders,
        max_epochs=max_epochs,
        config_dict={
            "device": device,
            "model_name": "jl_lstm_attention",
            "self_attention_config": self_attention_config,
        },
    )
```

### LSTM Self-Attention with Transformer Encoder
[Full Example]()
```python

```


## Citing & Authors

If you find this repository helpful, feel free to cite our publication [Hindi/Bengali Sentiment Analysis Using Transfer Learning and Joint Dual Input Learning with Self Attention](https://arxiv.org/abs/2202.05457):


```bibtex
@misc{khan2022hindibengali,
      title={Hindi/Bengali Sentiment Analysis Using Transfer Learning and Joint Dual Input Learning with Self Attention}, 
      author={Shahrukh Khan and Mahnoor Shahid},
      year={2022},
      eprint={2202.05457},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

