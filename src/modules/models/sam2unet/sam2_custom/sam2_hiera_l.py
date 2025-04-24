from .modeling.backbones.hieradet import Hiera
from .modeling.backbones.image_encoder import FpnNeck, ImageEncoder
from .modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from .modeling.memory_encoder import (
    CXBlock,
    Fuser,
    MaskDownSampler,
    MemoryEncoder,
)
from .modeling.position_encoding import PositionEmbeddingSine
from .modeling.sam2_base import SAM2Base
from .modeling.sam.transformer import RoPEAttention

SAM2_HIERA_L = SAM2Base(
    image_encoder=ImageEncoder(
        scalp=1,
        trunk=Hiera(
            embed_dim=144,
            num_heads=2,
            stages=[2, 6, 36, 4],
            global_att_blocks=[23, 33, 43],
            window_pos_embed_bkg_spatial_size=[7, 7],
            window_spec=[8, 4, 16, 8],
        ),
        neck=FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256,
                normalize=True,
                scale=None,
                temperature=10000,
            ),
            d_model=256,
            backbone_channel_list=[1152, 576, 288, 144],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        ),
    ),
    memory_attention=MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        layer=MemoryAttentionLayer(
            activation="relu",
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=False,
            self_attention=RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=[32, 32],
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1,
            ),
            d_model=256,
            pos_enc_at_cross_attn_keys=True,
            pos_enc_at_cross_attn_queries=False,
            cross_attention=RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=[32, 32],
                rope_k_repeat=True,
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1,
                kv_in_dim=64,
            ),
        ),
        num_layers=4,
    ),
    memory_encoder=MemoryEncoder(
        out_dim=64,
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=64,
            normalize=True,
            scale=None,
            temperature=10000,
        ),
        mask_downsampler=MaskDownSampler(
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        fuser=Fuser(
            layer=CXBlock(
                dim=256,
                kernel_size=7,
                padding=3,
                layer_scale_init_value=1e-6,
                use_dwconv=True,
            ),
            num_layers=2,
        ),
    ),
    num_maskmem=7,
    image_size=1024,
    sigmoid_scale_for_mem_enc=20.0,
    sigmoid_bias_for_mem_enc=-10.0,
    use_mask_input_as_output_without_sam=True,
    directly_add_no_mem_embed=True,
    use_high_res_features_in_sam=True,
    multimask_output_in_sam=True,
    iou_prediction_use_sigmoid=True,
    use_obj_ptrs_in_encoder=True,
    add_tpos_enc_to_obj_ptrs=False,
    only_obj_ptrs_in_the_past_for_eval=True,
    pred_obj_scores=True,
    pred_obj_scores_mlp=True,
    fixed_no_obj_ptr=True,
    multimask_output_for_tracking=True,
    use_multimask_token_for_obj_ptr=True,
    multimask_min_pt_num=0,
    multimask_max_pt_num=1,
    use_mlp_for_obj_ptr_proj=True,
    compile_image_encoder=False,
)
