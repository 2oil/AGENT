from enum import Enum
from adversarial_attacks import torchattacks


class AttackEnum(Enum):

    CM_BIM_001 = (torchattacks.CM_BIM, {"eps": 0.001, "alpha":0.0001, "steps" : 30 })
    CM_BIM_004 = (torchattacks.CM_BIM, {"eps": 0.004, "alpha":0.0004, "steps" : 30 })
    CM_BIM_008 = (torchattacks.CM_BIM, {"eps": 0.008, "alpha":0.0008, "steps" : 30 })
    CM_BIM_012 = (torchattacks.CM_BIM, {"eps": 0.012, "alpha":0.0012, "steps" : 30 })
    CM_BIM_016 = (torchattacks.CM_BIM, {"eps": 0.016, "alpha":0.0016, "steps" : 30 })

    ASV_BIM_001 = (torchattacks.ASV_BIM, {"eps": 0.001, "alpha":0.0001, "steps" : 30 })
    ASV_BIM_004 = (torchattacks.ASV_BIM, {"eps": 0.004, "alpha":0.0004, "steps" : 30 })
    ASV_BIM_008 = (torchattacks.ASV_BIM, {"eps": 0.008, "alpha":0.0008, "steps" : 30 })
    ASV_BIM_012 = (torchattacks.ASV_BIM, {"eps": 0.012, "alpha":0.0012, "steps" : 30 })
    ASV_BIM_016 = (torchattacks.ASV_BIM, {"eps": 0.016, "alpha":0.0016, "steps" : 30 })

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