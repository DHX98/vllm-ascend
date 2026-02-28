#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# Todo: Once https://github.com/vllm-project/vllm/pull/23553 is merged in vllm. Remove this model register.
import types

import torch
from vllm.logger import logger


def get_expert_map(self, layer_id):
    moe_model = get_eplb_moe_model(self)
    logger.debug("EPLB get_expert_map: layer_id=%d", layer_id)
    return moe_model.model.layers[layer_id].mlp.experts.expert_map


def get_log2phy_map(self, layer_id):
    moe_model = get_eplb_moe_model(self)
    logger.debug("EPLB get_log2phy_map: layer_id=%d", layer_id)
    return moe_model.model.layers[layer_id].mlp.experts.get_log2phy_map()


def get_all_expert_map(self, num_moe_layers):
    all_loads = []
    num_dense_layers = self.num_dense_layers if hasattr(
        self, "num_dense_layers") else 0
    for layer_id in range(num_moe_layers):
        load_tensor = self.get_expert_map(
            layer_id + num_dense_layers)  # (num_experts_per_layer,)
        all_loads.append(load_tensor)

    return torch.stack(all_loads, dim=0)


def get_all_moe_loads(self):
    moe_model = get_eplb_moe_model(self)
    num_dense_layers = self.num_dense_layers if hasattr(
        self, "num_dense_layers") else 0
    logger.debug(
        "EPLB get_all_moe_loads: num_moe_layers=%d, num_dense_layers=%d",
        self.num_moe_layers,
        num_dense_layers,
    )
    all_moe_loads = torch.stack(
        [moe_model.model.layers[layer_id + num_dense_layers].mlp.experts.moe_load \
            for layer_id in range(self.num_moe_layers)],
        dim=0
    )
    return all_moe_loads


def clear_all_moe_loads(self):
    moe_model = get_eplb_moe_model(self)
    num_dense_layers = self.num_dense_layers if hasattr(
        self, "num_dense_layers") else 0
    logger.debug(
        "EPLB clear_all_moe_loads: num_moe_layers=%d, num_dense_layers=%d",
        self.num_moe_layers,
        num_dense_layers,
    )
    for layer_id in range(self.num_moe_layers):
        moe_model.model.layers[layer_id +
                               num_dense_layers].mlp.experts.clear_moe_load()


def get_eplb_moe_model(model):
    if hasattr(model, "get_language_model"):
        language_model = model.get_language_model()
        if language_model is not None:
            logger.debug("EPLB resolved language model via get_language_model")
            return language_model
    if hasattr(model, "language_model"):
        logger.debug("EPLB resolved language model via language_model attr")
    return getattr(model, "language_model", model)


def model_register(model, model_config):
    model.get_expert_map = types.MethodType(get_expert_map, model)
    model.get_log2phy_map = types.MethodType(get_log2phy_map, model)
    model.get_all_expert_map = types.MethodType(get_all_expert_map, model)
    model.get_all_moe_loads = types.MethodType(get_all_moe_loads, model)
    model.clear_all_moe_loads = types.MethodType(clear_all_moe_loads, model)

    config = model_config.hf_text_config
    model_type = getattr(config, "model_type", "")
    text_config = getattr(config, "text_config", config)
    if model_type == "qwen3_vl_moe":
        # Compatibility for cases where hf_text_config falls back to top-level
        # multimodal config instead of text sub-config.
        config = text_config
        model_type = getattr(config, "model_type", model_type)

    if model_type in ("qwen3_moe", "qwen3_vl_moe_text"):
        model.num_dense_layers = getattr(config, "first_k_dense_replace", 0)
        model.num_moe_layers = config.num_hidden_layers - model.num_dense_layers
    elif model_type == "deepseek_v2" or model_type == "deepseek_v3":
        model.num_dense_layers = config.first_k_dense_replace
        model.num_moe_layers = config.num_hidden_layers - model.num_dense_layers
    else:
        raise NotImplementedError("EPLB is not supported.")

    # Keep wrapper model and inner language model EPLB metadata consistent.
    moe_model = get_eplb_moe_model(model)
    moe_model.num_dense_layers = model.num_dense_layers
    moe_model.num_moe_layers = model.num_moe_layers
    logger.info(
        "EPLB model_register done: model_type=%s, num_dense_layers=%d, "
        "num_moe_layers=%d, wrapped_language_model=%s",
        model_type,
        model.num_dense_layers,
        model.num_moe_layers,
        moe_model is not model,
    )
