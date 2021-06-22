__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Union, List, Any, Iterable, Dict

import torch
import torchvision.models.video as models

from jina import Executor, DocumentArray, requests


def _batch_generator(data: List[Any], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]


class VideoTorchEncoder(Executor):
    """
    Encode `Document` content from a ndarray, using the models from `torchvision.models`.
    :class:`VideoTorchEncoder` encodes content from a ndarray, potentially
    B x T x (Channel x Height x Width) into an ndarray of `B x D`.
    Internally, :class:`VideoTorchEncoder` wraps the models from
    `torchvision.models`: https://pytorch.org/docs/stable/torchvision/models.html
    :param model_name: the name of the model.
        Supported models include ``r3d_18``, ``mc3_18``, ``r2plus1d_18``
        Default is ``r3d_18``.
    """

    def __init__(self,
                 model_name: str = 'r3d_18',
                 device: Optional[str] = None,
                 default_batch_size: int = 32,
                 default_traversal_paths: Union[str, List[str]] = 'r',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths
        self.model = getattr(models, model_name)(pretrained=True).eval().to(self.device)

    def _get_embeddings(self, x) -> torch.Tensor:

        embeddings = torch.Tensor()

        def get_activation(model, model_input, output):
            nonlocal embeddings
            embeddings = output

        handle = self.model.avgpool.register_forward_hook(get_activation)
        self.model(x)
        handle.remove()
        return embeddings.flatten(1)

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Encode all docs with images and store the encodings in the embedding attribute of the docs.
        :param docs: documents sent to the encoder. The docs must have `blob` of the shape `Height x Width x 3`. By
            default, the input `blob` must be an `ndarray` with `dtype=uint8`. The `Height` and `Width` can have
            arbitrary values. When setting `use_default_preprocessing=False`, the input `blob` must have the size of
            `224x224x3` with `dtype=float32`.
        :param parameters: dictionary to define the `traversal_paths` and the `batch_size`. For example,
            `parameters={'traversal_paths': 'r', 'batch_size': 10}` will override the `self.default_traversal_paths` and
            `self.default_batch_size`.
        """
        if docs:
            document_batches_generator = self._get_input_data(docs, parameters)
            self._create_embeddings(document_batches_generator)

    def _get_input_data(self, docs: DocumentArray, parameters: dict):
        trav_paths = parameters.get('traversal_paths', self.default_traversal_paths)
        batch_size = parameters.get('batch_size', self.default_batch_size)

        # traverse thought all documents which have to be processed
        flat_docs = docs.traverse_flat(trav_paths)

        # filter out documents without images
        filtered_docs = [doc for doc in flat_docs if doc.blob is not None]

        return _batch_generator(filtered_docs, batch_size)

    def _create_embeddings(self, document_batches_generator: Iterable):
        with torch.no_grad():
            for document_batch in document_batches_generator:
                blob_batch = [d.blob for d in document_batch]
                tensor = torch.Tensor(blob_batch).to(self.device)
                embedding_batch = self._get_embeddings(tensor)
                numpy_embedding_batch = embedding_batch.cpu().numpy()
                for document, numpy_embedding in zip(document_batch, numpy_embedding_batch):
                    document.embedding = numpy_embedding
