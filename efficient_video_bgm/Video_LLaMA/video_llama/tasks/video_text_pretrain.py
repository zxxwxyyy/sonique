"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from efficient_video_bgm.Video_LLaMA.video_llama.common.registry import registry
from efficient_video_bgm.Video_LLaMA.video_llama.tasks.base_task import BaseTask


@registry.register_task("video_text_pretrain")
class VideoTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass