import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
)
from ..modules import init_graphormer_params, TokenGTGraphEncoder

@register_model("tokengt")
class TokenGTModel(FairseqEncoderModel):
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args
        self.apply(init_graphormer_params)
        self.encoder_embed_dim = args.encoder_embed_dim
    
    def max_nodes(self):
        return self.encoder.max_nodes

    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)


class TokenGTEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(dictionary=None)
        self.max_nodes = 128
        self.encoder_layers = args.encoder_layers
        self.num_attention_heads = args.encoder_attention_heads
        self.return_attention = args.return_attention

        self.graph_encoder = TokenGTGraphEncoder(
            
            num_atoms=args.num_atoms,
            num_edges=args.num_edges,
            
            rand_node_id=args.rand_node_id,
            rand_node_id_dim=args.rand_node_id_dim,
            orf_node_id=args.orf_node_id,
            orf_node_id_dim=args.orf_node_id_dim,           
            lap_node_id=args.lap_node_id,                       
            lap_node_id_k=args.lap_node_id_k, # whole: 454, clip: 50, baseline: 3, VidVRD: 30
            lap_node_id_sign_flip=args.lap_node_id_sign_flip,
            lap_node_id_eig_dropout=args.lap_node_id_eig_dropout,
            type_id=args.type_id,

            stochastic_depth=False,
            performer=False,
            performer_finetune=False,
            performer_nb_features=None,
            performer_feature_redraw_interval=1000,
            performer_generalized_attention=False,
            
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            
            encoder_normalize_before=False,
            layernorm_style='prenorm',
            apply_graphormer_init=False,
            activation_fn='gelu',
            return_attention=args.return_attention
        )

        self.embed_out = None
        self.lm_output_learned_bias = None
        self.share_input_output_embed = False

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(args, "remove_head", False)
        self.masked_lm_pooler = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.lm_head_transform_weight = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.activation_fn = utils.get_activation_fn('gelu')
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(args.num_output))    
            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(args.encoder_embed_dim, args.num_output, bias=False)
            else:
                raise NotImplementedError

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
        inner_states, graph_rep, attn_dict, node_mask = self.graph_encoder(batched_data, perturb=perturb)

        x = inner_states[-1].transpose(0, 1)  

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
        hidden_x = x
        
        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
                self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        
        elif self.embed_out is not None:
            x = self.embed_out(x)                  
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        # 객체 토큰만 골라내기.
        # x = (B, 2 + T, 26)
        object_idx = node_mask[:, 1:].int().sum(dim=1)   # except person node
           
        obj_bool = []
        for i in batched_data['node_per_frame']:
            if i != 0:
                obj_bool.append(False)
                obj_bool += [True]*(i-1) 
        out = x[0,2:2+batched_data['node_num'][0]][obj_bool]
        hidden_x = hidden_x[0,2:2+batched_data['node_num'][0]]
        
        return out, attn_dict, hidden_x

    def performer_finetune_setup(self):
        self.graph_encoder.performer_finetune_setup()

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes

    def upgrade_state_dict_named(self, state_dict, name):
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if "embed_out.weight" in k or "lm_output_learned_bias" in k:
                    del state_dict[k]
        return state_dict

