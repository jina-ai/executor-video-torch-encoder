__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina import Document, Flow
try:
    from video_torch_encoder import VideoTorchEncoder
except:
    from jinahub.encoder.video.video_torch_encoder import VideoTorchEncoder


def test_exec():
    f = Flow().add(uses=VideoTorchEncoder)
    with f:
        resp = f.post(on='/test', inputs=Document(), return_results=True)
        assert resp is not None
