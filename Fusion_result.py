from typing import Dict, Any, List, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import show

from models import (AudioVGGishModel, AudioL3Model, VisualResnetModel,
                    VisualL3Model, CombinerModel)
from utility.utilities import (FusionInfo, Features, FusionData,
                               create_model_and_features, load_model)
from utility.fusion_functions import (test_single_model,
                                      weighted_mean_rule_func, max_rule_func,
                                      product_rule_func,
                                      min_rule_func, jury_func,
                                      mode_rule_func, median_rule_func, deep_rule_func,
                                      test_combiner,
                                      custom_mode_rule_func, custom_median_rule_func,
                                      mean_rule_func, weighted_label_mean_rule_func)
from definitions import (BEST_AUDIO_VGGISH_MODEL,
                         BEST_VISUAL_RESNET_MODEL,
                         VISUAL_RESNET_TEST_FEATURES_FILE,
                         AUDIO_VGGISH_TEST_FEATURES_FILE,
                         LABELS, FRAMES_PER_VIDEO, VECTORS_PER_AUDIO,
                         BEST_VISUAL_L3_MODEL, BEST_AUDIO_L3_MODEL,
                         VISUAL_L3_TEST_FEATURES_FILE,
                         AUDIO_L3_TEST_FEATURES_FILE, VISUAL_RESNET_EMBED,
                         AUDIO_L3_EMBED, AUDIO_VGGISH_EMBED, VISUAL_L3_EMBED,
                         BEST_COMBINER_MODEL, MODELS_TEST_OUTPUTS_FILE)

visual_result = 0.773
audio_result = 0.569
fusion_result = 0.807

def mean_result(vis_re, aud_re, fusion_re):
    mean_re = (vis_re + aud_re + fusion_re) / 3
    return mean_re

def weight_result(vis_re, aud_re, fusion_re):
    all = vis_re + aud_re + fusion_re
    a = vis_re/all
    b = aud_re/all
    c = fusion_re/all
    weight_re = vis_re * a + aud_re * b + fusion_re * c
    return weight_re

def self_weight_result(vis_re, aud_re, fusion_re):
    self_weight_re = vis_re * 0.6 + aud_re * 0.2 + fusion_re * 0.2
    return self_weight_re


if __name__ == '__main__':

    mean = mean_result(visual_result, audio_result, fusion_result)
    weight = weight_result(visual_result, audio_result, fusion_result)
    self_weight = self_weight_result(visual_result, audio_result, fusion_result)

    # print("mean_result:", mean, "\nweight_result:", weight, "\nself_weight_result:", self_weight)
    print(f'mean_result:{mean:.3f}\nweight_result:{weight:.3f}\nself_weight_result:{self_weight:.3f}')
