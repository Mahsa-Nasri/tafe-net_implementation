""" Model configuration definition"""
from tafenet import *


def tafenet_default(emb_dim, feat_dim, feature_in_dim, hidden_dim, **kwargs):
    config = [2048, 2048, 2048]
    model = TAFENet(MetaLearner, config, feature_in_dim=feature_in_dim,
                    input_dim=emb_dim, feat_dim=feat_dim, hidden_dim=hidden_dim,
                    nhidden=3, return_emb=True, **kwargs)
    return model


def tafenet_2layer(emb_dim, feat_dim, feature_in_dim, hidden_dim, **kwargs):
    """Replication addition: 2-layer task embedding network T, which the paper
    (Sec. 4, 'Model configurations') uses for aPY to avoid overfitting."""
    config = [2048, 2048, 2048]
    model = TAFENet(MetaLearner, config, feature_in_dim=feature_in_dim,
                    input_dim=emb_dim, feat_dim=feat_dim, hidden_dim=hidden_dim,
                    nhidden=2, return_emb=True, **kwargs)
    return model
