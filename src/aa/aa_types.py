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

    ASV_PGD_0001 = (torchattacks.ASV_PGD, {"eps": 0.0001, "alpha":0.00001, "steps" : 200 })
    ASV_PGD_0003 = (torchattacks.ASV_PGD, {"eps": 0.0003, "alpha":0.00001, "steps" : 200 })
    ASV_PGD_0005 = (torchattacks.ASV_PGD, {"eps": 0.0005, "alpha":0.00001, "steps" : 200 })
    ASV_PGD_0007 = (torchattacks.ASV_PGD, {"eps": 0.0007, "alpha":0.00001, "steps" : 200 })

    NO_ATTACK = (None, {})

    @property
    def target_module(self):
        return self.name.split("_")[0]  # "CM" or "ASV"