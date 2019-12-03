{
  dataset_reader: {
    type: 'best2010_reader',
    token_indexers: {
      tokens: {
        type: 'single_id',
        namespace: 'tokens'
      }
    },
    lazy: false
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['tokens', 'num_tokens']],
    batch_size: 10
  },
  train_data_path: 'corpora/BEST2010_I2R/Train',
  validation_data_path: 'corpora/BEST2010_I2R/Dev',
  model: {
    type: 'ner_lstm',
    embedder: {
      tokens: {
        type: 'embedding',
        embedding_dim: 50,
        trainable: true
      }
    },
    encoder: {
      type: 'lstm',
      input_size: 50,
      hidden_size: 25,
      bidirectional: true
    }
  },
  trainer: {
    num_epochs: 40,
    patience: 10,
    cuda_device: -1,
    grad_clipping: 5.0,
    validation_metric: '-loss',
    optimizer: {
      type: 'adam',
      lr: 0.003
    }
  }
}
