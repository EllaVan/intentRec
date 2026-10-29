import torch
import logging
from torch import nn
from transformers import PretrainedConfig
from .__init__ import methods_map

__all__ = ['ModelManager']

class MIAConfig(PretrainedConfig):
    model_type = "mia"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MIA(nn.Module):
    def __init__(self, args):
        super(MIA, self).__init__()
        fusion_method = methods_map[args.method]
        self.args = args
        self.config = MIAConfig(**dict(args))
        self.model = fusion_method(args)

    def forward(self, text_feats, video_feats, audio_feats):

        video_feats, audio_feats = video_feats.float(), audio_feats.float()
        mm_model = self.model(text_feats, video_feats, audio_feats)

        return mm_model

class ModelManager:

    def __init__(self, args):

        self.logger = logging.getLogger(args.logger_name)
        self.device = args.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')
        self.model = self._set_model(args)

    def _set_model(self, args):

        model = MIA(args)
        model.to(self.device)
        return model