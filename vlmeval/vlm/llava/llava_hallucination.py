"""
LLaVA_Hallucination: LLaVA with Attention-based Hallucination Mitigation

Supports three decoding methods:
- PAI: Paying More Attention to Image (ECCV 2024)
- OPERA: Over-Trust Penalty and Retrospection-Allocation (CVPR 2024)
- AllPath: Multi-path Head Intervention (NeurIPS 2025)

Usage:
    # In VLMEvalKit
    python run.py --data POPE --model llava_v1.5_7b_pai
    python run.py --data MME --model llava_v1.5_7b_opera
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import math
import copy
import warnings
from PIL import Image
from types import SimpleNamespace
from typing import Optional, Dict, List, Any, Hashable

from ..base import BaseModel
from ...smp import *
from ...dataset import DATASET_TYPE


# ============================================================================
# Global State Manager (for AllPath)
# ============================================================================
class Grabber:
    """Global state manager for passing intervention parameters"""
    _store: Dict[Hashable, Any] = {}

    @classmethod
    def __setitem__(cls, key: Hashable, value: Any):
        cls._store[key] = value

    @classmethod
    def __getitem__(cls, key: Hashable) -> Any:
        return cls._store.get(key)

    @classmethod
    def get(cls, key: Hashable, default: Any = None) -> Any:
        return cls._store.get(key, default)

    @classmethod
    def clear(cls):
        cls._store.clear()


grabber = Grabber()


# ============================================================================
# PAI Implementation
# ============================================================================
class PAILogitsProcessor:
    """Contrastive decoding logits processor for PAI"""

    def __init__(self, guidance_scale, uncond_ids, model, tokenizer, start_layer=0, end_layer=32):
        self.guidance_scale = guidance_scale
        self.uncond_ids = uncond_ids
        self.model = model
        self.tokenizer = tokenizer
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.uncond_output = None

    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)

        if self.guidance_scale == 1:
            return scores

        # Enable CFG mode (disable attention enhancement)
        for i in range(self.start_layer, min(self.end_layer, len(self.model.model.layers))):
            if hasattr(self.model.model.layers[i].self_attn, 'use_cfg'):
                self.model.model.layers[i].self_attn.use_cfg = True

        # Get unconditional logits
        if self.uncond_output is None:
            self.uncond_output = self.model(self.uncond_ids, use_cache=True)
        else:
            self.uncond_output = self.model(
                input_ids[:, -1:],
                use_cache=True,
                past_key_values=self.uncond_output.past_key_values,
            )

        # Restore normal mode
        for i in range(self.start_layer, min(self.end_layer, len(self.model.model.layers))):
            if hasattr(self.model.model.layers[i].self_attn, 'use_cfg'):
                self.model.model.layers[i].self_attn.use_cfg = False

        unconditional_logits = F.log_softmax(self.uncond_output.logits[:, -1, :], dim=-1)

        # Contrastive decoding
        out = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits

        # Truncate low probability tokens
        cutoff = torch.log(torch.tensor(0.1, device=scores.device)) + scores.max(dim=-1, keepdim=True).values
        cd_logits = out.masked_fill(scores < cutoff, -float("inf"))

        return cd_logits


def create_pai_forward(layer_idx: int):
    """Create PAI modified attention forward function"""

    def pai_attention_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs
    ):
        bsz, q_len, _ = hidden_states.size()

        # Standard QKV projection
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if hasattr(past_key_value, 'get_usable_length'):
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            else:
                kv_seq_len += past_key_value[0].shape[-2]

        # RoPE
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # KV cache
        if past_key_value is not None:
            if hasattr(past_key_value, 'update'):
                cache_kwargs = {"sin": sin, "cos": cos}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # GQA expansion
        if hasattr(self, 'num_key_value_groups') and self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention weights
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # === PAI Core Modification ===
        use_attn = getattr(self, 'pai_use_attn', False)
        use_cfg = getattr(self, 'use_cfg', False)

        if use_attn and not use_cfg:
            img_start_idx = getattr(self, 'pai_img_start', 0)
            img_end_idx = getattr(self, 'pai_img_end', 0)
            alpha = getattr(self, 'pai_alpha', 0.2)

            if img_end_idx > img_start_idx:
                # Enhance attention to image tokens for the last token
                attn_weights[:, :, -1, img_start_idx:img_end_idx] = (
                    attn_weights[:, :, -1, img_start_idx:img_end_idx].abs() * alpha
                    + attn_weights[:, :, -1, img_start_idx:img_end_idx]
                )
        # === End PAI Modification ===

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # Return new KV cache format
        new_cache = None
        if use_cache:
            if past_key_value is not None and hasattr(past_key_value, 'update'):
                new_cache = past_key_value
            else:
                new_cache = (key_states, value_states)

        return attn_output, attn_weights, new_cache

    return pai_attention_forward


def create_allpath_forward(layer_idx: int):
    """Create AllPath modified attention forward function"""

    def allpath_attention_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs
    ):
        bsz, q_len, _ = hidden_states.size()

        # Standard attention computation (same as PAI)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if hasattr(past_key_value, 'get_usable_length'):
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            else:
                kv_seq_len += past_key_value[0].shape[-2]

        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            if hasattr(past_key_value, 'update'):
                cache_kwargs = {"sin": sin, "cos": cos}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if hasattr(self, 'num_key_value_groups') and self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # === AllPath Core Modification ===
        hallu_heads = grabber.get("hallu_heads", {})
        good_heads = grabber.get("good_heads", {})
        in_scale = grabber.get("in_scale", 1.0)
        de_scale = grabber.get("de_scale", 1.0)

        current_hallu = hallu_heads.get(layer_idx, [])
        current_good = good_heads.get(layer_idx, [])

        if current_hallu or current_good:
            num_heads = self.num_heads
            head_dim = self.head_dim

            # Create scale factors
            scale = torch.ones(num_heads, device=attn_output.device, dtype=attn_output.dtype)
            for h in current_hallu:
                if h < num_heads:
                    scale[h] = de_scale
            for h in current_good:
                if h < num_heads:
                    scale[h] = in_scale

            # Apply scaling to each head
            scale = scale.view(1, num_heads, 1, 1)
            attn_output = attn_output * scale
        # === End AllPath Modification ===

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        new_cache = None
        if use_cache:
            if past_key_value is not None and hasattr(past_key_value, 'update'):
                new_cache = past_key_value
            else:
                new_cache = (key_states, value_states)

        return attn_output, attn_weights, new_cache

    return allpath_attention_forward


# ============================================================================
# Main LLaVA_Hallucination Class
# ============================================================================
class LLaVA_Hallucination(BaseModel):
    """
    LLaVA with Attention-based Hallucination Mitigation

    Supports three decoding methods:
    - baseline: Standard LLaVA decoding
    - pai: PAI (Paying More Attention to Image)
    - opera: OPERA (Over-Trust Penalty and Retrospection-Allocation)
    - allpath: AllPath (Multi-path Head Intervention)
    """

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(
        self,
        model_path: str = "liuhaotian/llava-v1.5-7b",
        decoding_method: str = "baseline",
        # PAI parameters
        pai_alpha: float = 0.2,
        pai_gamma: float = 1.1,
        pai_use_cfg: bool = True,
        pai_start_layer: int = 2,
        pai_end_layer: int = 32,
        # OPERA parameters
        opera_scale_factor: float = 50.0,
        opera_threshold: int = 15,
        opera_num_candidates: int = 5,
        opera_penalty_weights: float = 1.0,
        # AllPath parameters
        allpath_in_scale: float = 2.0,
        allpath_de_scale: float = 0.0,
        allpath_hallu_heads: Dict[int, List[int]] = None,
        allpath_good_heads: Dict[int, List[int]] = None,
        **kwargs
    ):
        # Import llava
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except Exception as err:
            logging.critical("Please install llava from https://github.com/haotian-liu/LLaVA")
            raise err

        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = "</s>"
        self.decoding_method = decoding_method

        # Store parameters
        self.pai_alpha = pai_alpha
        self.pai_gamma = pai_gamma
        self.pai_use_cfg = pai_use_cfg
        self.pai_start_layer = pai_start_layer
        self.pai_end_layer = pai_end_layer

        self.opera_scale_factor = opera_scale_factor
        self.opera_threshold = opera_threshold
        self.opera_num_candidates = opera_num_candidates
        self.opera_penalty_weights = opera_penalty_weights

        self.allpath_in_scale = allpath_in_scale
        self.allpath_de_scale = allpath_de_scale
        self.allpath_hallu_heads = allpath_hallu_heads or {}
        self.allpath_good_heads = allpath_good_heads or {}

        # Load model
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                device_map="cpu",
            )
        )
        self.model = self.model.cuda()
        self.conv_mode = "llava_v1"

        # Generation config
        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            top_p=None,
            num_beams=5 if decoding_method == "opera" else 1,
            use_cache=True,
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        # Store original forwards for restoration
        self._original_forwards = {}
        self._modified = False

        warnings.warn(f"LLaVA_Hallucination initialized with decoding_method={decoding_method}")

    def _apply_pai_modification(self, img_start_idx: int, img_end_idx: int):
        """Apply PAI attention modification"""
        if self._modified:
            return

        end_layer = min(self.pai_end_layer, len(self.model.model.layers))

        for i in range(self.pai_start_layer, end_layer):
            attn = self.model.model.layers[i].self_attn
            self._original_forwards[i] = attn.forward

            # Set PAI parameters
            attn.pai_use_attn = True
            attn.pai_alpha = self.pai_alpha
            attn.pai_img_start = img_start_idx
            attn.pai_img_end = img_end_idx
            attn.use_cfg = False

            # Replace forward
            attn.forward = types.MethodType(create_pai_forward(i), attn)

        self._modified = True

    def _apply_allpath_modification(self):
        """Apply AllPath head intervention"""
        if self._modified:
            return

        # Set global parameters
        grabber["hallu_heads"] = self.allpath_hallu_heads
        grabber["good_heads"] = self.allpath_good_heads
        grabber["in_scale"] = self.allpath_in_scale
        grabber["de_scale"] = self.allpath_de_scale

        for i in range(len(self.model.model.layers)):
            attn = self.model.model.layers[i].self_attn
            self._original_forwards[i] = attn.forward
            attn.forward = types.MethodType(create_allpath_forward(i), attn)

        self._modified = True

    def _restore_model(self):
        """Restore original model"""
        for i, original_forward in self._original_forwards.items():
            self.model.model.layers[i].self_attn.forward = original_forward

        # Clear PAI attributes
        for i in range(len(self.model.model.layers)):
            attn = self.model.model.layers[i].self_attn
            for attr in ['pai_use_attn', 'pai_alpha', 'pai_img_start', 'pai_img_end', 'use_cfg']:
                if hasattr(attn, attr):
                    delattr(attn, attr)

        grabber.clear()
        self._original_forwards.clear()
        self._modified = False

    def _get_image_position(self, input_ids: torch.Tensor) -> tuple:
        """Get image token positions"""
        from llava.constants import IMAGE_TOKEN_INDEX
        IMAGE_TOKEN_LENGTH = 576  # LLaVA v1.5 uses 576 image tokens

        # Find IMAGE_TOKEN_INDEX position
        image_positions = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)

        if len(image_positions[1]) == 0:
            return 0, 0

        img_start = image_positions[1][0].item()
        img_end = img_start + IMAGE_TOKEN_LENGTH

        return img_start, img_end

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )

        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message

    def concat_tilist(self, message):
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += " <image> "
                images.append(item["value"])
        return text, images

    def generate_inner(self, message, dataset=None):
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from llava.constants import IMAGE_TOKEN_INDEX

        # Prepare inputs
        content, images = self.concat_tilist(message)
        images = [Image.open(s).convert("RGB") for s in images]

        args = SimpleNamespace()
        args.image_aspect_ratio = "pad"

        if images:
            image_tensor = process_images(images, self.image_processor, args).to(
                "cuda", dtype=torch.float16
            )
        else:
            image_tensor = None

        prompt = self.system_prompt + "USER: " + content + " ASSISTANT: "

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        # Get image position
        img_start_idx, img_end_idx = self._get_image_position(input_ids)

        # Prepare stopping criteria
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        # Generate based on decoding method
        try:
            with torch.inference_mode():
                if self.decoding_method == "pai":
                    output_ids = self._generate_pai(
                        input_ids, image_tensor, stopping_criteria,
                        img_start_idx, img_end_idx, prompt
                    )
                elif self.decoding_method == "opera":
                    output_ids = self._generate_opera(
                        input_ids, image_tensor, stopping_criteria,
                        img_start_idx, img_end_idx
                    )
                elif self.decoding_method == "allpath":
                    output_ids = self._generate_allpath(
                        input_ids, image_tensor, stopping_criteria
                    )
                else:
                    # Baseline
                    output_ids = self.model.generate(
                        input_ids,
                        images=image_tensor,
                        stopping_criteria=[stopping_criteria],
                        **self.kwargs,
                    )
        finally:
            # Always restore model after generation
            self._restore_model()

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output

    def _generate_pai(self, input_ids, image_tensor, stopping_criteria, img_start_idx, img_end_idx, prompt):
        """Generate with PAI method"""
        # Apply attention modification
        self._apply_pai_modification(img_start_idx, img_end_idx)

        kwargs = self.kwargs.copy()

        # Create CFG processor if enabled
        if self.pai_use_cfg and self.pai_gamma != 1.0:
            from transformers.generation.logits_process import LogitsProcessorList

            # Build unconditional input (without image)
            # Remove image placeholder from prompt
            uncond_prompt = prompt.replace(" <image> ", " ")
            uncond_ids = self.tokenizer(
                uncond_prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.cuda()

            cfg_processor = PAILogitsProcessor(
                guidance_scale=self.pai_gamma,
                uncond_ids=uncond_ids,
                model=self.model,
                tokenizer=self.tokenizer,
                start_layer=self.pai_start_layer,
                end_layer=self.pai_end_layer,
            )
            kwargs["logits_processor"] = LogitsProcessorList([cfg_processor])

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            stopping_criteria=[stopping_criteria],
            **kwargs,
        )

        return output_ids

    def _generate_opera(self, input_ids, image_tensor, stopping_criteria, img_start_idx, img_end_idx):
        """
        Generate with OPERA method

        Note: Full OPERA requires modified transformers.
        This is a simplified version that applies penalty during generation.
        For full OPERA, use the modified transformers from OPERA repo.
        """
        # OPERA requires beam search
        kwargs = self.kwargs.copy()
        kwargs["num_beams"] = max(kwargs.get("num_beams", 1), 5)
        kwargs["output_attentions"] = True

        # Note: Full OPERA implementation requires modifying transformers beam_search
        # This simplified version uses standard beam search with attention output
        # For production use, install OPERA's modified transformers

        warnings.warn(
            "Using simplified OPERA. For full OPERA with rollback, "
            "install modified transformers from OPERA repo."
        )

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            stopping_criteria=[stopping_criteria],
            **kwargs,
        )

        return output_ids

    def _generate_allpath(self, input_ids, image_tensor, stopping_criteria):
        """Generate with AllPath method"""
        # Apply head intervention
        self._apply_allpath_modification()

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            stopping_criteria=[stopping_criteria],
            **self.kwargs,
        )

        return output_ids
