"""Read-only original environment and checkpoint baseline probe."""
import json, os, platform, traceback, sys
from pathlib import Path
root = Path(__file__).resolve().parent
os.chdir(root / 'Our_experiment/HCSAC')
sys.path.insert(0, str(Path.cwd()))
result = {'python_version': platform.python_version(), 'flight_model_loaded': False, 'offload_model_loaded': False, 'inference_success': False, 'warnings': []}
try:
 import numpy as np
 import torch, torch_geometric
 from ENV.UAVenv_SAC_Original import UAVEnv, PolicyNet_CNN, PolicyNet_GCN
 result.update(torch_version=torch.__version__, torch_geometric_version=torch_geometric.__version__)
 torch.set_num_threads(4)
 env = UAVEnv(N=4)
 state = env.reset(seed=42)
 result['state_shape'] = list(state.shape)
 state, _, _ = env.step([0, 2, 4, 6])
 graph = env.get_obs_2()
 result.update(graph_nodes=int(graph.num_nodes), graph_edges=int(graph.num_edges), graph_node_types=graph.node_type.tolist())
 fly=PolicyNet_CNN(15,128,8)
 off=PolicyNet_GCN(6,128,5)
 fly.load_state_dict(torch.load('data/sac_model_fly.pt',map_location='cpu',weights_only=True)['actor_state_dict'])
 result['flight_model_loaded']=True
 off.load_state_dict(torch.load('data/sac_model_offload.pt',map_location='cpu',weights_only=True)['actor_state_dict'])
 result['offload_model_loaded']=True
 with torch.no_grad():
  result['flight_output_shape']=list(fly(torch.tensor(state[:1],dtype=torch.float32)).shape)
  result['offload_output_shape']=list(off(graph).shape)
 result['inference_success']=True
except Exception:
 result['error']=traceback.format_exc()
(root/'baseline_check.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result,indent=2))
