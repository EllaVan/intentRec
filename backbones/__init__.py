from .SubNets.FeatureNets import BERTEncoder
from .FusionNets.MINE import MINE

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder,
                    'bert-large-uncased': BERTEncoder
                }

methods_map = {
    'mine': MINE,
}