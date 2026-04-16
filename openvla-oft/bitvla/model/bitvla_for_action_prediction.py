from transformers import LlavaForConditionalGeneration,PretrainedConfig
try:
    from configuration_bit_vla import Bitvla_Config
except ImportError:
    import importlib
    Bitvla_Config = importlib.import_module("bitvla.configuration_bit_vla").Bitvla_Config
import numpy as np
import torch
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    NormalizationType,
)
from typing import Optional, Dict, Any,List,Tuple

from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast

from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)


class BitVLAForActionPrediction(LlavaForConditionalGeneration):
    config_class: PretrainedConfig = Bitvla_Config

    def __init__(self, config) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        self.vocab_size = self.config.vocab_size

    def set_constant(self, image_token_idx, proprio_pad_idx, ignore_idx, action_token_begin_idx, stop_index):
        self.image_token_idx = image_token_idx
        self.proprio_pad_idx = proprio_pad_idx
        self.action_token_begin_idx = action_token_begin_idx
        self.stop_index = stop_index
        self.ignore_idx = ignore_idx
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        proprio=None,
        proprio_projector=None,
        cache_position: Optional[torch.LongTensor] = None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        ) -> Tuple[int, LlavaCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0] # type: ignore

        # === Handle Multimodal Forward ===
        if (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during multimodal forward!"

            # Get input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)  # (B, seq_len, D)
            
            # change the vision padding to the real vision tokens
            if pixel_values is not None:
                vision_feature_layer = (
                    vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
                )
                vision_feature_select_strategy = (
                    vision_feature_select_strategy
                    if vision_feature_select_strategy is not None
                    else self.config.vision_feature_select_strategy
                )
                # pixel_values: b,num_images,c,h,w
                # for each image, we do self.get_image_features
                # then we concat the features of all images
                # pixel_values: (b,num_images,c,h,w) --> (b*num_images,c,h,w)
                b, num_images, c, h, w = pixel_values.shape
                pixel_values = pixel_values.view(-1, c, h, w)  # (b*num_images,c,h,w)
                image_embeds = self.get_image_features(
                    pixel_values = pixel_values,
                    vision_feature_layer = vision_feature_layer,
                    vision_feature_select_strategy = vision_feature_select_strategy,
                )
                
                # image_features: (b*num_images,seq_len,patch_size) --> (b*num_images*seq_len,patch_size)
                image_embeds = image_embeds.view(-1,image_embeds.shape[-1])
                n_image_tokens = (input_ids == self.image_token_idx).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.image_token_idx
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                
                
            # change the proprio padding to the real proprio tokens
            if proprio_projector is not None and proprio is not None:
                # proprio: (bsz, proprio_dim) or (propro_dim,)
                proprio = proprio.reshape(batch_size, -1)  # (bsz, proprio_dim)
                proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
                proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
                #（bsz, 1, llm_dim) --> (bsz*1, llm_dim)
                proprio_features = proprio_features.view(-1, proprio_features.shape[-1])
                n_proprio_tokens = (input_ids == self.proprio_pad_idx).sum().item()
                n_proprio_features = proprio_features.shape[0]
                if n_proprio_tokens != n_proprio_features:
                    raise ValueError(
                        f"Proprio features and proprio tokens do not match: tokens: {n_proprio_tokens}, features {n_proprio_features}"
                    )
                
                mask = input_ids == self.proprio_pad_idx
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                proprio_mask = mask_expanded.to(inputs_embeds.device)
                
                proprio_features = proprio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(proprio_mask, proprio_features)
                
            
            # Extract action masks
            # Action tokens are those in labels that are not ignore, not newline, and not end-of-sequence tokens
            all_actions_mask = (labels != self.ignore_idx) & (labels != self.stop_index)
          
            # Replace the embeddings of the action tokens with zeros
            # (Later on, the positional embeddings will be added to them)
            all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
            inputs_embeds = inputs_embeds * ~all_actions_mask 
            outputs = LlavaForConditionalGeneration.forward(
                self,
                input_ids = None,
                attention_mask=attention_mask,
                position_ids=None,
                pixel_values=None,
                labels=labels,
                inputs_embeds=inputs_embeds,
                past_key_values=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_bi_attn = True, # use bi-directional attention in oft
            )
        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        return outputs

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """Prepares input for action prediction by adding necessary tokens"""
        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * self.stop_index
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        """Creates labels tensor for action prediction if not provided"""
        # Extend labels tensor with fake action labels
        ARBITRARY_ACTION_TOKEN_IDX = self.action_token_begin_idx + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # Replace last label token with stop token
        labels[:, -1] = self.stop_index

        return labels

    def _process_action_masks(self, labels):
        """Helper to get action masks from labels"""
        current_action_mask = get_current_action_mask(labels,ignore_index=self.ignore_idx,action_token_begin_idx=self.action_token_begin_idx)
        next_actions_mask = get_next_actions_mask(labels,ignore_index=self.ignore_idx,action_token_begin_idx=self.action_token_begin_idx)
        all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
        return all_actions_mask
        
    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )

        return actions
    
    def _regression_or_discrete_prediction(
        self,
        input_ids,
        input_embeddings,
        all_actions_mask,
        attention_mask,
        labels,
        action_head=None,
        pixel_values = None,
    ):
        """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""
        # Zero out action token embeddings
        all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
        input_embeddings = input_embeddings * ~all_actions_mask

        llava_output = LlavaForConditionalGeneration.forward(
                self,
                input_ids = None,
                attention_mask=attention_mask,
                position_ids=None,
                pixel_values=None,
                labels=None,
                inputs_embeds=input_embeddings,
                past_key_values=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_bi_attn = True, # use bi-directional attention in oft
            )
        all_actions_mask = self._process_action_masks(labels[:,1:])
        # Extract hidden states for action tokens
        last_hidden_states = llava_output.hidden_states[-1]  # (B, seq_len, D)
        last_hidden_states = last_hidden_states[:, : -1, :]  # (B, act_chunk_len, D)
        # Use the action mask to extract the hidden states of the actions
        actions_hidden_states = last_hidden_states[all_actions_mask.squeeze(-1)].unsqueeze(0) # (B, act_chunk_len, D)

        # Handle different prediction methods
        if action_head is not None:
            # L1 regression prediction
            normalized_actions = action_head.predict_action(actions_hidden_states)
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:
            # Discrete token-based prediction
            predicted_action_token_ids = (
                llava_output.logits[all_actions_mask.squeeze(-1)].unsqueeze(0)
                .argmax(dim=2)
                .cpu()
                .numpy()
            )
            # FIXME: We do not support discrete action prediction right now
            # It seems that vocab_size here is not correct. This should be the dimension of the logit layer, which is actually larger than the vocab_size in the tokenizer. What we actually need here is the vocab_size from the tokenizer.
            discretized_actions = self.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return normalized_actions, actions_hidden_states
    
    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        **kwargs: str,
    ) -> np.ndarray:
        """Predict actions from input sequence, with options for different prediction methods.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalization statistics
            proprio: Proprioceptive features
            proprio_projector: Projector for proprioceptive features
            action_head: Optional head for L1 regression prediction
            **kwargs: Additional arguments including pixel_values and attention_mask

        Returns:
            Tuple of (unnormalized_actions, action_hidden_states)
        """
        pixel_values = kwargs["pixel_values"]
        attention_mask = kwargs["attention_mask"]

        # Create fake labels tensor (needed for action mask)
        labels = input_ids.clone()
        labels[:] = self.ignore_idx

        # Prepare inputs by adding necessary tokens
        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

        # Update labels tensor for action mask computation later
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)

        # Get input embeddings and action masks
        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)
        
        # vision tokens
        if pixel_values is not None:
            vision_feature_layer = (
                vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
            )
            vision_feature_select_strategy = (
                vision_feature_select_strategy
                if vision_feature_select_strategy is not None
                else self.config.vision_feature_select_strategy
            )
            # pixel_values: b,num_images,c,h,w
            # for each image, we do self.get_image_features
            # then we concat the features of all images
            # pixel_values: (b,num_images,c,h,w) --> (b*num_images,c,h,w)
            b, num_images, c, h, w = pixel_values.shape
            pixel_values = pixel_values.view(-1, c, h, w)  # (b*num_images,c,h,w)
            image_embeds = self.get_image_features(
                pixel_values = pixel_values,
                vision_feature_layer = vision_feature_layer,
                vision_feature_select_strategy = vision_feature_select_strategy,
            )
            
            # image_features: (b*num_images,seq_len,patch_size) --> (b*num_images*seq_len,patch_size)
            image_embeds = image_embeds.view(-1,image_embeds.shape[-1])
            n_image_tokens = (input_ids == self.image_token_idx).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.image_token_idx
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(input_embeddings)
            image_mask = mask_expanded.to(input_embeddings.device)

            image_embeds = image_embeds.to(input_embeddings.device, input_embeddings.dtype)
            input_embeddings = input_embeddings.masked_scatter(image_mask, image_embeds)

        # Add proprioceptive features if provided
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            batch_size = input_ids.shape[0] if input_ids is not None else input_embeddings.shape[0] # type: ignore
            proprio = torch.Tensor(proprio).to(input_embeddings.device, dtype=input_embeddings.dtype)
            if proprio_projector is not None and proprio is not None:
                # proprio: (bsz, proprio_dim) or (propro_dim,)
                proprio = proprio.reshape(batch_size, -1)  # (bsz, proprio_dim)
                proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
                proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
                #（bsz, 1, llm_dim) --> (bsz*1, llm_dim)
                proprio_features = proprio_features.view(-1, proprio_features.shape[-1])
                n_proprio_tokens = (input_ids == self.proprio_pad_idx).sum().item()
                n_proprio_features = proprio_features.shape[0]
                if n_proprio_tokens != n_proprio_features:
                    raise ValueError(
                        f"Proprio features and proprio tokens do not match: tokens: {n_proprio_tokens}, features {n_proprio_features}"
                    )
                
                mask = input_ids == self.proprio_pad_idx
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(input_embeddings)
                proprio_mask = mask_expanded.to(input_embeddings.device)
                
                proprio_features = proprio_features.to(input_embeddings.device, input_embeddings.dtype)
                input_embeddings = input_embeddings.masked_scatter(proprio_mask, proprio_features)

        # Run regression or discrete token-based prediction
        normalized_actions, actions_hidden_states = self._regression_or_discrete_prediction(
            input_ids,
            input_embeddings,
            all_actions_mask,
            attention_mask,
            labels,
            action_head,
            pixel_values,
        )

        # Unnormalize predicted actions
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)

        return actions, actions_hidden_states

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        """Validate and resolve the unnormalization key for action statistics"""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["min"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
        