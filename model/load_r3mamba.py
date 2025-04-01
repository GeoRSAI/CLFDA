from model.UNetFormer import UNetFormer
from model.RS3Mamba import RS3Mamba, load_pretrained_ckpt
from model.rs_mamba_ss import RSM_SS
from model.CLFDA_rsm_ss import RSM_SS_OURS

def unetformer(num_classes=15, num_channels=3, use_bn=False):
    model = UNetFormer(num_classes=num_classes).cuda()
    return model

def rs3mamba(num_classes=15, num_channels=3, use_bn=False):
    model = RS3Mamba(num_classes=num_classes).cuda()
    model = load_pretrained_ckpt(model)
    return model


def rsm_ss(num_classes=16, num_channels=3, use_bn=False):
    dims = [96, 192, 384, 768]
    depths = [2, 2, 9, 2]
    ssm_d_state = 16
    ssm_dt_rank = "auto"
    ssm_ratio = 2.0
    mlp_ratio = 4.0
    downsample_version = "v3"
    patchembed_version =  "v2"
    model = RSM_SS(num_classes=num_classes,
                 dims=dims,
                 depths=depths,
                 ssm_d_state=ssm_d_state,
                 ssm_dt_rank=ssm_dt_rank,
                 ssm_ratio=ssm_ratio,
                 mlp_ratio=mlp_ratio,
                 downsample_version=downsample_version,
                 patchembed_version=patchembed_version).cuda()
    return model

def CLFDA(num_classes=16, num_channels=3, use_bn=False):
    dims = [96, 192, 384, 768]
    depths = [2, 2, 9, 2]
    ssm_d_state = 16
    ssm_dt_rank = "auto"
    ssm_ratio = 2.0
    mlp_ratio = 4.0
    downsample_version = "v3"
    patchembed_version =  "v2"
    model = RSM_SS_OURS(num_classes=num_classes,
                 dims=dims,
                 depths=depths,
                 ssm_d_state=ssm_d_state,
                 ssm_dt_rank=ssm_dt_rank,
                 ssm_ratio=ssm_ratio,
                 mlp_ratio=mlp_ratio,
                 downsample_version=downsample_version,
                 patchembed_version=patchembed_version).cuda()
    return model
