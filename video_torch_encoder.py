__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, List, Any, Iterable, Dict

import torch
import torch.nn as nn

import torchvision.models.video as models
from torchvision import transforms

from jina import Executor, DocumentArray, requests


# https://github.com/pytorch/vision/blob/d391a0e992a35d7fb01e11110e2ccf8e445ad8a0/references/video_classification/transforms.py#L13

class ConvertFHWCtoFCHW(nn.Module):

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(0, 3, 1, 2)


class ConvertFCHWtoCFHW(nn.Module):

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


def _batch_generator(data: List[Any], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]


class VideoTorchEncoder(Executor):
    """
    Encode `Document` content, using the models from `torchvision.models`.
    :class:`VideoTorchEncoder` encodes content from Documents containing video blobs

    Internally, :class:`VideoTorchEncoder` wraps the models from
    `torchvision.models`: https://pytorch.org/docs/stable/torchvision/models.html
    :param model_name: the name of the model.
        Supported models include ``r3d_18``, ``mc3_18``, ``r2plus1d_18``
        Default is ``r3d_18``.
    :param use_default_preprocessing: if True, the same preprocessing is used which got used during training
        - prevents training-serving gap.
    :param device: device to use for encoding ['cuda', 'cpu] - if not set, the device is detected automatically
    :param default_batch_size: fallback batch size in case there is not batch size sent in the request
    :param default_traversal_paths: fallback traversal path in case there is not traversal path sent in the request
    """

    def __init__(self,
                 model_name: str = 'r3d_18',
                 use_default_preprocessing: bool = True,
                 device: Optional[str] = None,
                 default_batch_size: int = 32,
                 default_traversal_paths: List[str] = ['r'],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths
        self.model = getattr(models, model_name)(pretrained=True).eval().to(self.device)
        self.use_default_preprocessing = use_default_preprocessing
        if self.use_default_preprocessing:
            # https://github.com/pytorch/vision/blob/master/references/video_classification/train.py
            # Eval preset transformation
            mean = (0.43216, 0.394666, 0.37645)
            std = (0.22803, 0.22145, 0.216989)
            resize_size = (128, 171)
            crop_size = (112, 112)
            self.transforms = transforms.Compose([
                ConvertFHWCtoFCHW(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(resize_size),
                transforms.Normalize(mean=mean, std=std),
                transforms.CenterCrop(crop_size),
                ConvertFCHWtoCFHW()
            ])

    def _get_embeddings(self, x) -> torch.Tensor:
        embeddings = None
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
        :param docs: documents sent to the encoder. The docs must have `blob` of the shape `Channels x NumFrames x Height x Width`
        with `Height` and `Width` equals to 112, if no default preprocessing is requested. When setting
        `use_default_preprocessing=True`, the input `blob` must have the size of `Frame x Height x Width x Channel`.
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
                if self.use_default_preprocessing:
                    tensors = [self.transforms(torch.Tensor(d.blob).to(dtype=torch.uint8)) for d in document_batch]
                    tensor = torch.stack(tensors).to(self.device)
                else:
                    tensor = torch.stack([torch.Tensor(d.blob) for d in document_batch]).to(self.device)
                embedding_batch = self._get_embeddings(tensor)
                numpy_embedding_batch = embedding_batch.cpu().numpy()
                for document, numpy_embedding in zip(document_batch, numpy_embedding_batch):
                    document.embedding = numpy_embedding
