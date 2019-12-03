import torch
import torch.nn as nn

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.models import Model

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from typing import Dict, Optional

@Model.register('ner_lstm')
class NerLSTM(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder) -> None:
        super().__init__(vocab)

        self._embedder = embedder
        self._encoder = encoder
        self._classifier = nn.Linear(in_features=encoder.get_output_dim(),
                                     out_features=vocab.get_vocab_size('ne_tags'))

        self._f1 = SpanBasedF1Measure(vocab, 'ne_tags')

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                ne_tags: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)

        embedded = self._embedder(tokens)
        encoded = self._encoder(embedded, mask)
        classified = self._classifier(encoded)

        self._f1(classified, ne_tags, mask)

        output: Dict[str, torch.Tensor] = {}

        if ne_tags is not None:
            output["loss"] = sequence_cross_entropy_with_logits(classified, ne_tags, mask)

        return output

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return self._f1.get_metric(reset)
