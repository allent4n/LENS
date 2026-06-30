import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
import sys
import kornia
from copy import deepcopy

sys.path.append("./clip4caption")
from modules.tokenization import BertTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CaptionGenerator
from train import collect_hypothesis_and_scores, collate_active_info, beam_decode_step, get_inst_idx_to_tensor_position_map
from modules.beam import Beam
import math
from transformers.modeling_outputs import Seq2SeqLMOutput,BaseModelOutput

from transformers import BartTokenizer, BartForConditionalGeneration, \
    PegasusForConditionalGeneration, PegasusTokenizer, \
    LEDForConditionalGeneration, LEDTokenizer, \
    LongT5ForConditionalGeneration, AutoTokenizer
from peft import LoraConfig
from peft import get_peft_model


MODEL_NAME = 'facebook/bart-large-cnn'
# google/pegasus-large, google/pegasus-pubmed, allenai/led-large-16384, allenai/led-base-16384, google/long-t5-tglobal-base


class MultiModalFusion(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        
        self.video_gate = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Sigmoid()
        )
        
        self.audio_gate = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Sigmoid()
        )
        
        
        self.modal_attention = nn.MultiheadAttention(hidden_size, 8, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        self.gate_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.attention_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        #self._reset_parameters()
    """
    def _reset_parameters(self):

        for name, p in self.named_parameters():
            print(f"name, p :{name,p}")
            if 'weight' in name:
                if 'norm' not in name:  
                    nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
    """      

    def _safe_normalize(self, x, dim=-1, eps=1e-8):
        norms = torch.norm(x.clone(), p=2, dim=dim, keepdim=True)
        norms = torch.where(norms > eps, norms, torch.ones_like(norms) * eps)
        return x.clone() / norms
        
    def _compute_gate(self, features, gate_module):
        
        if torch.isnan(features).any():
            raise ValueError("Input features contain NaN")
            
        try:
            
            norm_features = self._safe_normalize(features)
            
            
            gate = gate_module(norm_features)
            
            
            gate = torch.clamp(gate * self.gate_scale, min=0.0, max=1.0)
            
            if torch.isnan(gate).any():
                raise ValueError("Gate computation produced NaN")
                
            return gate
            
        except Exception as e:
            print(str(e))
            raise
            
    def forward(self, text_features, video_features, audio_features):

        try:
            
            for name, feat in [("text", text_features), ("video", video_features), ("audio", audio_features)]:
                if torch.isnan(feat).any():
                    raise ValueError(f"{name} features contain NaN")
            
            
            text_features = self._safe_normalize(text_features)
            # print(f"text feats in fusion:{text_features.shape}")
            #dnduis:torch.Size([32, 1024, 768])
            video_features = self._safe_normalize(video_features)
            audio_features = self._safe_normalize(audio_features)
            # print(f"video feats in fusion: {video_features.shape}")
            # print(f"audio feats in fusion: {audio_features.shape}")
            
            
            video_gate = self._compute_gate(video_features, self.video_gate)
            audio_gate = self._compute_gate(audio_features, self.audio_gate)
            
            
            video_gated = video_features * video_gate
            audio_gated = audio_features * audio_gate
            
            
            video_attended, _ = self.modal_attention(
                video_gated.transpose(0, 1),
                text_features.transpose(0, 1),
                text_features.transpose(0, 1)
            )
            video_attended = video_attended.transpose(0, 1) * self.attention_scale
            
            
            audio_attended, _ = self.modal_attention(
                audio_gated.transpose(0, 1),
                text_features.transpose(0, 1),
                text_features.transpose(0, 1)
            )
            audio_attended = audio_attended.transpose(0, 1) * self.attention_scale
            
            
            video_attended = self.norm2(video_attended)
            audio_attended = self.norm2(audio_attended)
            
            
            combined = torch.cat([text_features, video_attended, audio_attended], dim=-1)
            fused = self.fusion_layer(combined)
            output = self.norm3(fused + text_features)
            
            
            if torch.isnan(output).any():
                raise ValueError("Fusion produced NaN output")
                
            return output
            
        except Exception as e:
            print(f"Error in fusion forward pass: {str(e)}")
            raise
            
    def reset_gates(self):
        with torch.no_grad():
            self.gate_scale.fill_(0.1)
            self.attention_scale.fill_(0.1)


class AttentiveMemory(nn.Module):
    def __init__(self, hidden_size, memory_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        
        self.memory_query = nn.Linear(hidden_size, hidden_size)
        self.memory_key = nn.Linear(hidden_size, hidden_size)
        self.memory_value = nn.Linear(hidden_size, hidden_size)
        
        
        self.Wu1 = nn.Linear(hidden_size, hidden_size)  
        self.Wu2 = nn.Linear(hidden_size, hidden_size)
        self.Wg1 = nn.Linear(hidden_size, hidden_size)
        self.Wg2 = nn.Linear(hidden_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.memory_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, memory_state, input_features):
        if memory_state is not None:
            memory_state = memory_state.detach()

        batch_size = input_features.size(0)
        
        if memory_state is None:
            memory_state = torch.zeros(
                batch_size, self.memory_size, self.hidden_size,
                device=input_features.device
            )
        
        
        queries = self.memory_query(memory_state)
        keys = self.memory_key(input_features) 
        values = self.memory_value(input_features)
        
        
        keys = keys.detach()
        values = values.detach()
        

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)
        attention_weights = F.softmax(attention_scores, dim=-1)
        S_t = torch.matmul(attention_weights, values)
        
        
        U_t = torch.tanh(self.Wu1(memory_state) + self.Wu2(S_t))
        G_t = torch.sigmoid(self.Wg1(memory_state) + self.Wg2(S_t))
        
        
        new_memory = G_t * U_t + (1 - G_t) * memory_state
        new_memory = self.layer_norm(new_memory)
        
        memory_output, _ = self.memory_attn(
            input_features,  
            memory_state,    
            memory_state     
        )
        enhanced_features = self.output_layer_norm(input_features + memory_output)
 
        return new_memory, enhanced_features


class CompressiveMemory(nn.Module):
    def __init__(self, hidden_size, memory_size, compression_rate=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.compression_rate = compression_rate
        
        
        self.compressed_size = memory_size // 2
        self.uncompressed_size = memory_size - self.compressed_size

        self.compress = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=compression_rate,
            stride=compression_rate,
            padding=compression_rate // 2  
        )
        
        self.memory_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, memory_state, input_features):
        target_len = input_features.size(1)
        batch_size = input_features.size(0)
        
        # 首先处理 memory_state 为 None 的情况
        if memory_state is None:
            memory_state = torch.zeros(
                batch_size, self.memory_size, self.hidden_size,
                device=input_features.device
            )
        else:

            if memory_state.size(0) != batch_size:

                new_memory_state = torch.zeros(
                    batch_size, memory_state.size(1), memory_state.size(2),
                    device=memory_state.device
                )

                min_batch = min(batch_size, memory_state.size(0))
                new_memory_state[:min_batch] = memory_state[:min_batch]
                memory_state = new_memory_state


            if memory_state.size(1) > target_len:
                memory_state = memory_state[:, :target_len, :]
            elif memory_state.size(1) < target_len:
                pad = torch.zeros(
                    memory_state.size(0), 
                    target_len - memory_state.size(1), 
                    memory_state.size(2), 
                    device=memory_state.device
                )
                memory_state = torch.cat([memory_state, pad], dim=1)
            
            memory_state = memory_state.detach()

        seq_len = input_features.size(1)
        M_c = memory_state[:, :self.compressed_size]
        M_u = memory_state[:, self.compressed_size:]
        
        
        if seq_len < self.compression_rate:
            
            M_u_new = input_features
            compressed = M_c
        else:
            # print(f"shape M_u, input_features: {M_u.shape, input_features.shape}")
            M_u_new = torch.cat([M_u, input_features.detach()], dim=1)
            
            
            to_compress = M_u_new[:, :(-self.uncompressed_size)]
            to_compress = to_compress.transpose(1, 2)  
            
            try:
                compressed = self.compress(to_compress)
                compressed = compressed.transpose(1, 2)  
            except RuntimeError:
                
                compressed = to_compress.mean(dim=2, keepdim=True).transpose(1, 2)
        
        
        M_u = M_u_new[:, -self.uncompressed_size:]
        
        
        M_c = torch.cat([M_c, compressed], dim=1)[:, -self.compressed_size:]
        
        
        new_memory = torch.cat([M_c, M_u], dim=1)
        
        
        memory_output, _ = self.memory_attn(
            input_features,      
            new_memory,         
            new_memory          
        )
        
        
        enhanced_features = self.output_layer_norm(input_features + memory_output)
        
        return new_memory, enhanced_features


class MumSUM(BartForConditionalGeneration):
    def __init__(self, config, args=None):
        super().__init__(config)
        # if torch.cuda.is_available():
        #     self.cuda()

        self.tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

        self.memory_size = 1024
        self.hidden_size = config.d_model
        self.video_memory_states = {}

        self.encoder_memory = AttentiveMemory(
            hidden_size=config.d_model,
            memory_size=1024
        ).to('cuda')
        
        self.decoder_memory = CompressiveMemory(
            hidden_size=config.d_model,
            memory_size=1024
        ).to('cuda')

        self.register_buffer('encoder_memory_state', None)
        self.register_buffer('decoder_memory_state', None)
        
        
        self.video_projection = nn.Sequential(
            nn.Linear(1024, config.d_model),  
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout if hasattr(config, 'dropout') else config.dropout_rate)
        )
        
        self.audio_projection = nn.Sequential(
            nn.Linear(384, config.d_model), 
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout if hasattr(config, 'dropout') else config.dropout_rate)
        )

        self.modality_fusion = MultiModalFusion(
            hidden_size=config.d_model,
            dropout=config.dropout if hasattr(config, 'dropout') else config.dropout_rate
        )
        self.contrastive_head = nn.Linear(config.d_model, 128)

    
    @property
    def device(self):
        return next(self.parameters()).device
            
    def forward(self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        video_features=None, 
        audio_features=None,
        video_name=None,
        video_mask=None,
        moment_mask=None,
        encoder_memory_state=None,
        decoder_memory_state=None,
        **kwargs
    ):
        device = self.device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        #print(f"memory state size: {len(self.video_memory_states)}")


        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        
        text_features = outputs.encoder_last_hidden_state

        if video_features is not None and audio_features is not None and video_mask is not None and moment_mask is not None:
            # print(f"input_ids exists: {input_ids != None}")
            # print(f"video or audio feats exists: {video_features != None and audio_features != None}")
            # print(f"moment and video mask check: {moment_mask, video_mask}")
            
            combined_mask = (video_mask & moment_mask).float()
            mask_for_features = combined_mask.unsqueeze(-1)
            
            masked_video_features = video_features * mask_for_features
            masked_audio_features = audio_features * mask_for_features

            batch_size = video_features.shape[0]

            encoder_memory, decoder_memory = self._init_memory_for_video(video_name, batch_size=batch_size, device=device)
            self.encoder_memory_state = encoder_memory
            self.decoder_memory_state = decoder_memory

            video_hidden = self.video_projection(masked_video_features)
            audio_hidden = self.audio_projection(masked_audio_features)

            target_len = text_features.shape[1]

            video_pad = nn.functional.pad(
                video_hidden,
                (0, 0, 0, target_len - video_hidden.shape[1])
            )
            audio_pad = nn.functional.pad(
                audio_hidden,
                (0, 0, 0, target_len - audio_hidden.shape[1])
            )

            multimodal_features = self.modality_fusion(
                text_features,
                video_pad,
                audio_pad
            )
        else:
            multimodal_features = text_features

        if self.encoder_memory is not None:
            self.encoder_memory_state, enhanced_features = self.encoder_memory(self.encoder_memory_state, multimodal_features)
        multimodal_features = enhanced_features

        if self.decoder_memory is not None and outputs.decoder_hidden_states is not None:
            decoder_features = outputs.decoder_hidden_states[-1]
            self.decoder_memory_state, enhanced_decoder = self.decoder_memory(
                self.decoder_memory_state,
                decoder_features
            )
            
            outputs.decoder_hidden_states = outputs.decoder_hidden_states[:-1] + (enhanced_decoder,)



        if labels is not None:
            # base_loss = outputs.loss if hasattr(outputs, 'loss') else None
            # scaled_features = multimodal_features * 0.1
            # contrastive_features = self.contrastive_head(scaled_features)
            # contrastive_loss = self.compute_contrastive_loss(contrastive_features, labels)
            # total_loss = base_loss + 0.01 * contrastive_loss

            h = BaseModelOutput(
            last_hidden_state=multimodal_features,
            hidden_states=outputs.decoder_hidden_states,
            attentions=None)

            out = super().forward(
            encoder_outputs=h,
            labels=labels)

            total_loss = out['loss']
        else:
            total_loss = None

        if video_name is not None:
            self._update_memory_for_video(
                video_name,
                self.encoder_memory_state,
                self.decoder_memory_state
            )

        outputs = Seq2SeqLMOutput(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=multimodal_features,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
        return outputs
        




    def reset_memory_for_video(self, video_id):
        self.video_memory_states[video_id] = {
            'encoder_memory': None,
            'decoder_memory': None
        }


    def _init_memory_for_video(self, video_names, batch_size, device):
        if isinstance(video_names, str):
            video_names = [video_names] * batch_size
            
        encoder_memories = []
        decoder_memories = []
        
        for video_name in video_names:
            if video_name not in self.video_memory_states:
                self.video_memory_states[video_name] = {
                    'encoder_memory': None,
                    'decoder_memory': None
                }
            
            memory_state = self.video_memory_states[video_name]
            
            
            if memory_state['encoder_memory'] is None:
                memory_state['encoder_memory'] = torch.zeros(
                    1, 1024, self.hidden_size,  
                    device=device
                )
            if memory_state['decoder_memory'] is None:
                memory_state['decoder_memory'] = torch.zeros(
                    1, 1024, self.hidden_size,  
                    device=device
                )
                
            
            encoder_mem = memory_state['encoder_memory']
            decoder_mem = memory_state['decoder_memory']
            
            
            if encoder_mem.size(0) != 1:
                encoder_mem = encoder_mem[:1]
            if decoder_mem.size(0) != 1:
                decoder_mem = decoder_mem[:1]
            
            if encoder_mem.size(1) == 1024:
                encoder_memories.append(encoder_mem.detach())
            if decoder_mem.size(1) == 1024:
                decoder_memories.append(decoder_mem.detach())
        
        # for i in decoder_memories:
        #     print(f"shape of memories: {i.shape}")

        
        batch_encoder_memory = torch.zeros(
            batch_size, 1024, self.hidden_size,  
            device=device
        )

        batch_decoder_memory = torch.zeros(
            batch_size, 1024, self.hidden_size,  
            device=device
        )
        if len(encoder_memories) != 0:
            batch_encoder_memory = torch.cat(encoder_memories, dim=0)

        if len(decoder_memories) != 0:
            batch_decoder_memory = torch.cat(decoder_memories, dim=0)
        
        return batch_encoder_memory, batch_decoder_memory


    def _update_memory_for_video(self, video_names, encoder_memory, decoder_memory):
        for i, video_name in enumerate(video_names):
            
            self.video_memory_states[video_name] = {
                'encoder_memory': encoder_memory[i:i+1].detach(),
                'decoder_memory': decoder_memory[i:i+1].detach()
            }




    def compute_contrastive_loss(self, features, labels):
        if labels is None:

            return torch.zeros(1, device=features.device, requires_grad=True)
        
        try:    
            
            features = features.mean(dim=1)  
            
            features_norm = torch.norm(features, p=2, dim=-1, keepdim=True)
            features = features / (features_norm + 1e-8)  
            
            
            sim_matrix = torch.matmul(features, features.t())  
            
            
            sequence_labels = torch.mode(labels, dim=1)[0]  
            pos_mask = (sequence_labels.unsqueeze(0) == sequence_labels.unsqueeze(1)).float()  
            pos_mask.fill_diagonal_(0)  
            
            
            pos_mask = pos_mask + 1e-8
            pos_mask = pos_mask / pos_mask.sum(1, keepdim=True)  
            
            
            temperature = 0.07
            logits = sim_matrix / temperature  
            
            
            log_probs = F.log_softmax(logits, dim=1)  
            
            
            loss = -torch.sum(log_probs * pos_mask, dim=1).mean()  
            
            return loss
                
        except Exception as e:
            print(f"Error in contrastive loss computation: {str(e)}")
            return torch.zeros(1, device=features.device, requires_grad=True)

    def generate(self, *args, **kwargs):

        device = self.device
        

        if args and torch.is_tensor(args[0]):
            args = list(args)
            args[0] = args[0].to(device)
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                kwargs[key] = value.to(device)


        for key in ['video_features', 'audio_features', 'video_mask', 'moment_mask']:
            kwargs.pop(key, None)

        return super().generate(*args, **kwargs)

def generate_summary(model, batch):
    print("generate_summary is called")
    device = model.device
    prompts = batch['prompts']
    dummy_texts = [prompt for prompt in prompts]
    

    # tokenizer_name = os.path.join(current_dir, "..", "bart-base")
    tokenizer = model.tokenizer

    # Process all texts at once
    inputs = tokenizer(dummy_texts,
                       max_length=1024,
                       truncation=True,
                       padding=True,
                       return_tensors="pt")
    
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    video_features = batch['vis_feats'].to(device)
    audio_features = batch['asr_feats'].to(device)
    video_mask = batch['vis_mask'].to(device)
    moment_mask = batch['moment_mask'].to(device)
    video_names = batch['video_fnames']
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            video_features=video_features,
            audio_features=audio_features,
            video_mask=video_mask,
            moment_mask=moment_mask,
            video_name=video_names,
            attention_mask=attention_mask,
            max_length=128,
            min_length=0,
            length_penalty=2.0,
            num_beams=5,
            synced_gpus=False,
            early_stopping=True
        )
    
    summaries = [model.tokenizer.decode(g, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True)
                for g in outputs]
    return summaries



class MomentModel(nn.Module):

    def __init__(self, n_frames=-1, asr_dim=-1, args=None):
        super(MomentModel, self).__init__()

        self.args = args
        self.n_frames = n_frames ## -1

        embed_dim = 512
        
        self.asr_dim = asr_dim ## 512
        self.use_asr = asr_dim > 0 ## True if > 0, else False
        if self.use_asr:
            ## 512 -> 512
            self.asr_enc_layer = nn.Sequential(
                nn.LayerNorm(asr_dim),
                nn.Linear(asr_dim, embed_dim)
            )

        ## map timestamp to embedding
        ## scalar in [0, 1] -> 512
        self.temporal_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 0: frames outside of moment
        # 1: frames inside of moment
        self.mask_embed = nn.Embedding(2, embed_dim)

        self.boundary_embed = nn.Embedding(2, embed_dim)

        dropout = 0.1
        self.input_dropout = nn.Dropout(dropout)

        # Moment Retrieval


        # kernel_size = 5
        # padding = kernel_size // 2


        self.moment_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=3,
                padding=1,
            )
        )

        # Moment Segmentation

        embed_dim_2 = 768
        
        self.start_predictor = nn.Sequential(
            # nn.LayerNorm(embed_dim),
            # nn.Linear(embed_dim, embed_dim),
            # nn.GELU(),
            nn.Linear(embed_dim_2, 1),
        )

        self.end_predictor = nn.Sequential(
            # nn.LayerNorm(embed_dim),
            # nn.Linear(embed_dim, embed_dim),
            # nn.GELU(),
            nn.Linear(embed_dim_2, 1),
        )

        self.segment_predictor = nn.Sequential(
            # nn.LayerNorm(embed_dim),
            # nn.Linear(embed_dim, embed_dim),
            # nn.GELU(),
            nn.Linear(embed_dim_2, 1),
        )

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        model_state_dict = torch.load("./pretrained_weights/clip4caption_vit-b-32_model.bin", map_location='cpu')
        args.d_model = embed_dim
        args.video_dim = embed_dim
        args.max_frames = args.max_frames_step_captioning
        
        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        
        ## The model is here 
        self.clip4cap_model = CaptionGenerator.from_pretrained("bert-base-uncased", "visual-base", "audio-base", "decoder-base",
                            cache_dir=cache_dir, state_dict=model_state_dict, task_config=args, max_position_embeddings_override=2048)
        
        self.clip_g_map = nn.Linear(1024, embed_dim)
        self.clip_g_map_text = nn.Linear(1024, embed_dim)

        sys.path.append("./EVA_clip")
        from eva_clip import build_eva_model_and_transforms
        self.clip_model, self.clip_preprocess = build_eva_model_and_transforms("EVA_CLIP_g_14", pretrained="./pretrained_weights/eva_clip_psz14.pt")
        print("Loaded EVA CLIP G")

        self.clip_model = self.clip_model.float()
        self.clip_model.eval()

        self.freeze_clip()

        self.memsum_model = MumSUM.from_pretrained(MODEL_NAME, args)

        for param in self.memsum_model.parameters():
            param.requires_grad = False
        
        for module in [self.memsum_model.video_projection, self.memsum_model.audio_projection, self.memsum_model.modality_fusion, 
                       self.memsum_model.contrastive_head, self.memsum_model.encoder_memory, self.memsum_model.decoder_memory]:
            for param in module.parameters():
                param.requires_grad = True

        language_lora_config = LoraConfig(
            peft_type="LORA",
            r=16,  # rank of the update matrices
            lora_alpha=16,  # scaling factor for LoRA updates
            target_modules=["q", "v", "lm_head", "shared"],  # apply LoRA to query and value matrices
            lora_dropout=0.1,  # dropout rate for LoRA updates
            bias="none",  # do not train bias parameters
                #modules_to_save=["lm_head"]  # also train the classifier parameters
        )
        self.memsum_model = get_peft_model(self.memsum_model, language_lora_config)


    def encode_text_sliding_window(self, clip_text_ids, window_size=77, stride=38):
        device = clip_text_ids.device
        batch_size = clip_text_ids.size(0)
        text_feats = []

        for i in range(batch_size):
            input_ids = clip_text_ids[i]
            non_pad_mask = (input_ids != 0)
            actual_length = non_pad_mask.sum().item()
            actual_tokens = input_ids[:actual_length]

            if actual_length == 0:
                # Handle empty input
                text_feat = torch.zeros(self.clip_model.text_projection.size(1), device=device)
                text_feats.append(text_feat)
                continue

            if actual_length <= window_size:
                # Process as single chunk
                chunk_input_ids = actual_tokens
                if actual_length < window_size:
                    pad = torch.zeros(window_size - actual_length, dtype=torch.long, device=device)
                    chunk_input_ids = torch.cat([chunk_input_ids, pad])
                chunk_attention_mask = (chunk_input_ids != 0).long()
                # Encode
                chunk_feats = self.clip_model.encode_text(
                    chunk_input_ids.unsqueeze(0)
                ).float()
                text_feats.append(chunk_feats.squeeze(0))
                continue

            # Split into overlapping chunks
            chunks_input_ids = []
            chunks_attention_mask = []
            start = 0
            while start < actual_length:
                end = start + window_size
                chunk_tokens = actual_tokens[start:end]
                chunk_length = len(chunk_tokens)
                # Pad if necessary
                if chunk_length < window_size:
                    pad = torch.zeros(window_size - chunk_length, dtype=torch.long, device=device)
                    chunk = torch.cat([chunk_tokens, pad])
                else:
                    chunk = chunk_tokens
                # Create attention mask
                mask = torch.zeros(window_size, dtype=torch.long, device=device)
                mask[:chunk_length] = 1
                chunks_input_ids.append(chunk)
                chunks_attention_mask.append(mask)
                start += stride

            # Convert to tensors
            chunks_input_ids = torch.stack(chunks_input_ids)  # [num_chunks, window_size]
            chunks_attention_mask = torch.stack(chunks_attention_mask)  # [num_chunks, window_size]

            # Encode all chunks in one batch
            chunk_feats = self.clip_model.encode_text(
                chunks_input_ids
            ).float()  # [num_chunks, embed_dim]

            # Aggregate features
            avg_feats = chunk_feats.mean(dim=0)
            text_feats.append(avg_feats)

        text_feat = torch.stack(text_feats, dim=0)  # [batch_size, embed_dim]
        return text_feat


    def freeze_clip(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

    def train_step(self, batch):
        task = batch['tasks'][0]

        if task == 'moment_retrieval':
            return self.train_moment_retrieval(batch)
        elif task == 'memsum':
            return self.train_memsum(batch)
        else:
            raise NotImplementedError

    def test_step(self, batch, **kwargs):
        task = batch['tasks'][0]

        if task == 'moment_retrieval':
            return self.test_moment_retrieval(batch, **kwargs)

        elif task == 'memsum':
            return self.test_memsum(batch, **kwargs)
        else:
            raise NotImplementedError

    def foward_moment_shared(self, video_feats, text_feat, video_mask=None, moment_mask=None, asr_feats=None, boundary_mask=None):
        '''
        The goal is to align these different input modalities (video, text, and audio) into a common feature space 
        where they can be compared and used together to predict the moment boundaries.
        1. Aligns the video, text, and ASR features together using learned embeddings.
        2. Adds temporal embeddings to the video features to provide information about the relative time position of each frame.
        3. Adds mask embeddings to focus the model on the frames that are considered important for the moment.
        '''
        B, max_n_frames, embed_dim = video_feats.size()

        video_feats = self.clip_g_map(video_feats)
        text_feat = self.clip_g_map_text(text_feat)
        
        video_feats = self.clip4cap_model.normalize_video(video_feats)

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        feats = video_feats * text_feat.unsqueeze(1)

        if self.use_asr:
            asr_feats = self.asr_enc_layer(asr_feats)
            feats += asr_feats

        if boundary_mask is not None:
            boundary_emb = self.boundary_embed(boundary_mask)
            feats += boundary_emb

        # [batch_size]
        if video_mask is None:
            video_mask = torch.ones((B, max_n_frames), device=video_feats.device, dtype=torch.long)
        n_frames_batch = video_mask.sum(dim=-1).long()

        # time representation normlized in [-1, 1]
        normalized_times = []
        max_n_frames = max(n_frames_batch)
        for n_frames in n_frames_batch:

            # [0, 1] -> [-0.5, 0.5] ->  [-1, 1]
            normalized_time = (torch.linspace(0, 1, n_frames) - 0.5) * 2

            n_pad = max_n_frames - n_frames
            padding = torch.zeros(n_pad)
            normzlied_time = torch.cat([normalized_time, padding]).view(1, max_n_frames, 1)
            normalized_times.append(normzlied_time)

        normalized_times = torch.cat(normalized_times, dim=0).to(video_feats.device)

        temporal_embed = self.temporal_embed(normalized_times)
        feats += temporal_embed

        mask_embed = self.mask_embed(moment_mask)
        feats += mask_embed

        assert video_mask.dim() == 2, video_mask.shape
        extended_attention_mask = video_mask[:, None, None, :]

        dtype = feats.dtype
        extended_attention_mask = extended_attention_mask.to(dtype=dtype) 
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        
        feats = self.clip4cap_model.get_visual_output(feats, torch.zeros((B, max_n_frames)).long().to(feats.device), shaped=True)

        return feats

    def forward_moment_retrieval(self, video_feats, text_feat, video_mask=None, moment_mask=None, asr_feats=None):
        '''
        This method processes the video, text, and optional audio features (ASR) and predicts where the moment starts and ends.
        '''

        B, max_n_frames, embed_dim = video_feats.size()

        ## get visual output from the clip4caption model
        feats = self.foward_moment_shared(video_feats, text_feat, video_mask, moment_mask=moment_mask, asr_feats=asr_feats)

        start_logits = self.start_predictor(feats).squeeze(2)
        end_logits = self.end_predictor(feats).squeeze(2)

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
        }

    def train_moment_retrieval(self, batch):
        ## Get all the features 
        device = next(self.parameters()).device
        video_feats = batch['vis_feats'].to(device)

        video_mask = batch['vis_mask'].to(device)
        moment_mask = batch['moment_mask'].to(device)

        start_target = batch['moment_retrieval_start_target'].to(device)
        end_target = batch['moment_retrieval_end_target'].to(device)

        asr_feats = None
        if self.use_asr:
            asr_feats = batch['asr_feats'].to(device)

        with torch.no_grad():
            clip_text_ids = batch['clip_text_ids'].to(device)
            # text_feat = self.clip_model.encode_text(clip_text_ids).float()
            text_feat = self.encode_text_sliding_window(clip_text_ids)

        if video_feats.shape[1] > 2048:
            video_feats = video_feats[:, :2048, :]
            
        ## Pass the feature value to forward_moment_retrieval to get the start and end logits. 
        out = self.forward_moment_retrieval(
            video_feats, text_feat, video_mask=video_mask, moment_mask=moment_mask, asr_feats=asr_feats)
        start_logits = out['start_logits']
        end_logits = out['end_logits']
        
        ## Building the start and end label/target
        ## _start_target and _end_target are tensors initialized to zeros, with the same shape as start_logits and end_logits
        _start_target = torch.zeros(start_logits.size(), device=start_logits.device)
        _end_target = torch.zeros(end_logits.size(), device=end_logits.device)
        
        ## This function modifies the _start_target and _end_target tensors in-place, 
        ## setting the value 1 at the positions indicated by start_target and end_target.
        _start_target.scatter_(1, start_target.unsqueeze(1), 1)
        _end_target.scatter_(1, end_target.unsqueeze(1), 1)

        start_loss = F.binary_cross_entropy_with_logits(start_logits, _start_target, reduction='none')
        end_loss = F.binary_cross_entropy_with_logits(end_logits, _end_target, reduction='none')
        
        ## moment_mask: which frames in the video are valid for moment retrieval
        ## multiplication: ensure that the loss is computed only for the valid frames.
        start_loss = start_loss * moment_mask
        end_loss = end_loss * moment_mask
        
        ## xx.sum(): summed across all frames and across all videos in the batch
        ## Division: Normalizing by valid frames, clamp(min=1) is to avoid divided by 0
        start_loss = start_loss.sum() / moment_mask.sum().clamp(min=1)
        end_loss = end_loss.sum() / moment_mask.sum().clamp(min=1)

        loss = (start_loss + end_loss) / 2

        result = {
            'loss': loss,
        }

        return result

    @torch.no_grad()
    def test_moment_retrieval(self, batch, **kwargs):
        device = next(self.parameters()).device
        video_feats = batch['vis_feats'].to(device)

        video_mask = batch['vis_mask'].to(device)
        moment_mask = batch['moment_mask'].to(device)

        if self.use_asr:
            asr_feats = batch['asr_feats'].to(device)

        # text_feat = batch['text_feat'].to(device)
        with torch.no_grad():
            clip_text_ids = batch['clip_text_ids'].to(device)
            # text_feat = self.clip_model.encode_text(clip_text_ids).float()
            text_feat = self.encode_text_sliding_window(clip_text_ids)

        out = self.forward_moment_retrieval(
            video_feats, text_feat, video_mask=video_mask, moment_mask=moment_mask, asr_feats=asr_feats)

        start_logits = out['start_logits']
        end_logits = out['end_logits']

        start_logits[video_mask == 0] = -1e10
        end_logits[video_mask == 0] = -1e10

        start = start_logits.argmax(dim=1)
        end = end_logits.argmax(dim=1)


        pred_boundaries = torch.stack([start, end], dim=-1)

        start_target = batch['moment_retrieval_start_target']
        end_target = batch['moment_retrieval_end_target']

        result = {
            'prediction': pred_boundaries.detach().tolist(),
        }

        return result

    def train_memsum(self, batch):
        # print("train_memsum is running")
        device = next(self.parameters()).device

        video_feats = batch['vis_feats'].to(device)
        video_mask = batch['vis_mask'].to(device) 
        moment_mask = batch['moment_mask'].to(device)
        prompts = batch['prompts']
        label_summaries = batch['target_text']
        video_names = batch['video_fnames']


        input_temp = [f"The video shows {prompt}" for prompt in prompts]
        labels_temp = [label for label in label_summaries]
        inputs = self.memsum_model.tokenizer(
            input_temp,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        labels = self.memsum_model.tokenizer(
            labels_temp,
            max_length=128,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        asr_feats = batch['asr_feats'].to(device)

        # print(f"asr feats: {asr_feats}")

        # print("starts memsum forward")
        """
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        video_features=None, 
        audio_features=None,
        video_name=None,
        **kwargs
        """
        outputs = self.memsum_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels.input_ids,
            video_features=video_feats,
            audio_features=asr_feats,
            video_name=video_names,
            video_mask=video_mask,
            moment_mask=moment_mask
        )
        
        loss = outputs.loss
        # print(f"loss is:{loss}")

        result = {
            'loss': loss,
        }
        
        return result

    @torch.no_grad()
    def test_memsum(self, batch, **kwargs):
        device = next(self.parameters()).device

        model = self.memsum_model
        model.eval()

        generated_text = generate_summary(model, batch)
        print(generated_text,flush=True)
        
        return {'prediction': generated_text}
