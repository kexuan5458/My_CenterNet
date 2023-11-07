import dotsi

model_cfg = dict(
    CLASS_AGNOSTIC=True,
    SHARED_FC=[256, 256],
    CLS_FC=[256, 256],
    REG_FC=[256, 256],
    DP_RATIO=0.3,

    TARGET_CONFIG=dict(
        ROI_PER_IMAGE=128,
        FG_RATIO=0.5,
        SAMPLE_ROI_BY_EACH_CLASS=True,
        CLS_SCORE_TYPE='roi_iou',
        CLS_FG_THRESH=0.75,
        CLS_BG_THRESH=0.25,
        CLS_BG_THRESH_LO=0.1,
        HARD_BG_RATIO=0.8,
        REG_FG_THRESH=0.55
    ),
    LOSS_CONFIG=dict(
        CLS_LOSS='BinaryCrossEntropy',
        REG_LOSS='L1',
        LOSS_WEIGHTS={
            'rcnn_cls_weight': 1.0,
            'rcnn_reg_weight': 1.0,
            'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
    )
)

dot_dict = dotsi.Dict(model_cfg)
import ipdb; ipdb.set_trace(context=7)
