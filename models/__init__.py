from models.tf2torch_ens3_adv_inc_v3 import KitModel as IncV3Ens3
from models.tf2torch_ens4_adv_inc_v3 import KitModel as IncV3Ens4
from models.tf2torch_ens_adv_inc_res_v2 import KitModel as IncResV2Ens
from models.tf2torch_adv_inception_v3 import KitModel as AdvIncV3

ens_models = {'IncV3Ens3': IncV3Ens3, 
          'IncV3Ens4': IncV3Ens4, 
          'IncResV2Ens': IncResV2Ens,
          'AdvIncV3': AdvIncV3}

ens_path = {'IncV3Ens3': 'models/npy/tf2torch_ens3_adv_inc_v3.npy', 
        'IncV3Ens4': 'models/npy/tf2torch_ens4_adv_inc_v3.npy', 
        'IncResV2Ens': 'models/npy/tf2torch_ens_adv_inc_res_v2.npy'}