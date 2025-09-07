from enum import Enum
from adversarial_attacks import torchattacks


class AttackEnum(Enum):

    CM_FGSM_0001 = (torchattacks.CM_FGSM, {"eps": 0.0001})
    CM_FGSM_0003 = (torchattacks.CM_FGSM, {"eps": 0.0003})
    CM_FGSM_0005 = (torchattacks.CM_FGSM, {"eps": 0.0005})
    CM_FGSM_0007 = (torchattacks.CM_FGSM, {"eps": 0.0007})

    CM_BIM_0001 = (torchattacks.CM_BIM, {"eps": 0.0001, "alpha":0.00001, "steps" : 200 })
    CM_BIM_0003 = (torchattacks.CM_BIM, {"eps": 0.0003, "alpha":0.00001, "steps" : 200 })
    CM_BIM_0005 = (torchattacks.CM_BIM, {"eps": 0.0005, "alpha":0.00001, "steps" : 200 })
    CM_BIM_0007 = (torchattacks.CM_BIM, {"eps": 0.0007, "alpha":0.00001, "steps" : 200 })


    CM_BIM_001 = (torchattacks.CM_BIM, {"eps": 0.001, "alpha":0.0001, "steps" : 30 })
    CM_BIM_004 = (torchattacks.CM_BIM, {"eps": 0.004, "alpha":0.0004, "steps" : 30 })
    CM_BIM_008 = (torchattacks.CM_BIM, {"eps": 0.008, "alpha":0.0008, "steps" : 30 })
    CM_BIM_012 = (torchattacks.CM_BIM, {"eps": 0.012, "alpha":0.0012, "steps" : 30 })
    CM_BIM_016 = (torchattacks.CM_BIM, {"eps": 0.016, "alpha":0.0016, "steps" : 30 })
    
    CM_PGD_0001 = (torchattacks.CM_PGD, {"eps": 0.0001, "alpha":0.00001, "steps" : 200 })
    CM_PGD_0003 = (torchattacks.CM_PGD, {"eps": 0.0003, "alpha":0.00001, "steps" : 200 })
    CM_PGD_0005 = (torchattacks.CM_PGD, {"eps": 0.0005, "alpha":0.00001, "steps" : 200 })
    CM_PGD_0007 = (torchattacks.CM_PGD, {"eps": 0.0007, "alpha":0.00001, "steps" : 200 })

    ASV_FGSM_0001 = (torchattacks.ASV_FGSM, {"eps": 0.0001})
    ASV_FGSM_0003 = (torchattacks.ASV_FGSM, {"eps": 0.0003})
    ASV_FGSM_0005 = (torchattacks.ASV_FGSM, {"eps": 0.0005})
    ASV_FGSM_0007 = (torchattacks.ASV_FGSM, {"eps": 0.0007})

    ASV_BIM_0001 = (torchattacks.ASV_BIM, {"eps": 0.0001, "alpha":0.00001, "steps" : 200 })
    ASV_BIM_0003 = (torchattacks.ASV_BIM, {"eps": 0.0003, "alpha":0.00001, "steps" : 200 })
    ASV_BIM_0005 = (torchattacks.ASV_BIM, {"eps": 0.0005, "alpha":0.00001, "steps" : 200 })
    ASV_BIM_0007 = (torchattacks.ASV_BIM, {"eps": 0.0007, "alpha":0.00001, "steps" : 200 })

    ASV_BIM_001 = (torchattacks.ASV_BIM, {"eps": 0.001, "alpha":0.0001, "steps" : 30 })
    ASV_BIM_004 = (torchattacks.ASV_BIM, {"eps": 0.004, "alpha":0.0004, "steps" : 30 })
    ASV_BIM_008 = (torchattacks.ASV_BIM, {"eps": 0.008, "alpha":0.0008, "steps" : 30 })
    ASV_BIM_012 = (torchattacks.ASV_BIM, {"eps": 0.012, "alpha":0.0012, "steps" : 30 })
    ASV_BIM_016 = (torchattacks.ASV_BIM, {"eps": 0.016, "alpha":0.0016, "steps" : 30 })

    ASV_PGD_0001 = (torchattacks.ASV_PGD, {"eps": 0.0001, "alpha":0.00001, "steps" : 200 })
    ASV_PGD_0003 = (torchattacks.ASV_PGD, {"eps": 0.0003, "alpha":0.00001, "steps" : 200 })
    ASV_PGD_0005 = (torchattacks.ASV_PGD, {"eps": 0.0005, "alpha":0.00001, "steps" : 200 })
    ASV_PGD_0007 = (torchattacks.ASV_PGD, {"eps": 0.0007, "alpha":0.00001, "steps" : 200 })

    BOTH_DD_004 = (torchattacks.DD, {"eps": 0.004, "alpha":0.0004, "steps" : 30 })
    BOTH_DD_008 = (torchattacks.DD, {"eps": 0.008, "alpha":0.0008, "steps" : 30 })

    BOTH_AGENT_001 = (torchattacks.AGENT, {"eps": 0.001, "alpha":0.0001, "steps" : 30 })
    BOTH_AGENT_004 = (torchattacks.AGENT, {"eps": 0.004, "alpha":0.0004, "steps" : 30 })
    BOTH_AGENT_008 = (torchattacks.AGENT, {"eps": 0.008, "alpha":0.0008, "steps" : 30 })
    BOTH_AGENT_012 = (torchattacks.AGENT, {"eps": 0.012, "alpha":0.0012, "steps" : 30 })
    BOTH_AGENT_016 = (torchattacks.AGENT, {"eps": 0.016, "alpha":0.0016, "steps" : 30 })

    BOTH_ReLU_00 = (torchattacks.ReLU, {"eps": 0, "alpha": 0, "steps" : 1 })
    BOTH_ReLU_004 = (torchattacks.ReLU, {"eps": 0.004, "alpha":0.0004, "steps" : 30 })
    BOTH_ReLU_008 = (torchattacks.ReLU, {"eps": 0.008, "alpha":0.0008, "steps" : 30 })

    NO_ATTACK = (None, {})

    @property
    def target_module(self):
        return self.name.split("_")[0]  # "CM" or "ASV"