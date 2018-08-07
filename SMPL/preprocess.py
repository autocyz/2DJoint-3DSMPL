import numpy as np
import pickle
import sys



def backwards_compatibility_replacements(dd):

    # replacements
    if 'default_v' in dd:
        dd['v_template'] = dd['default_v']
        del dd['default_v']
    if 'template_v' in dd:
        dd['v_template'] = dd['template_v']
        del dd['template_v']
    if 'joint_regressor' in dd:
        dd['J_regressor'] = dd['joint_regressor']
        del dd['joint_regressor']
    if 'blendshapes' in dd:
        dd['posedirs'] = dd['blendshapes']
        del dd['blendshapes']
    if 'J' not in dd:
        dd['J'] = dd['joints']
        del dd['joints']

    # defaults
    if 'bs_style' not in dd:
        dd['bs_style'] = 'lbs'



def ready_arguments(fname_or_dict):

    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict, 'wb'))
    else:
        dd = fname_or_dict
        
    backwards_compatibility_replacements(dd)
        
    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1]*3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas'])+dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:,0])        
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:,1])        
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:,2])        
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T    
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
    else:    
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
            
    return dd


output_path = './m_lbs.pkl'

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Error: expected source model path.')
    exit(-1)
  src_path = sys.argv[1]
  src_data = ready_arguments(src_path)
  
  model = {
    'J_regressor': src_data['J_regressor'],
    'weights': np.array(src_data['weights']),
    'posedirs': np.array(src_data['posedirs']),
    'v_template': np.array(src_data['v_template']),
    'shapedirs': np.array(src_data['shapedirs']),
    'f': np.array(src_data['f']),
    'kintree_table': src_data['kintree_table']
  }
  with open(output_path, 'wb') as f:
    pickle.dump(model, f)
