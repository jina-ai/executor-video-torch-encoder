__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import pytest
import numpy as np
from jina import Document, DocumentArray

try:
    from video_torch_encoder import VideoTorchEncoder
except:
    from jinahub.encoder.video.video_torch_encoder import VideoTorchEncoder


@pytest.mark.parametrize('model_name', ['r3d_18', 'mc3_18', 'r2plus1d_18'])
def test_video_torch_encoder(model_name):
    ex = VideoTorchEncoder(model_name=model_name)
    da = DocumentArray([Document(blob=np.random.random((3, 2, 112, 112))) for _ in range(10)])
    ex.encode(da, {})
    assert len(da) == 10
    for doc in da:
        assert doc.embedding.shape == (512,)


@pytest.mark.parametrize('batch_size', [1, 3, 10])
def test_video_torch_encoder_traversal_paths(batch_size):
    ex = VideoTorchEncoder()

    def _create_doc_with_video_chunks():
        d = Document(blob=np.random.random((3, 2, 112, 112)))
        d.chunks = [Document(blob=np.random.random((3, 2, 112, 112))) for _ in range(5)]
        return d

    da = DocumentArray([_create_doc_with_video_chunks() for _ in range(10)])
    ex.encode(da, {'traversal_paths': ['r', 'c'], 'batch_size': batch_size})
    assert len(da) == 10
    for doc in da:
        assert doc.embedding.shape == (512,)
        assert len(doc.chunks) == 5
        for chunk in doc.chunks:
            assert chunk.embedding.shape == (512,)
