import torch
import numpy as np

if __name__ == '__main__':
    # best_model = torch.load('./model/3head_t_rnn_v_linear_cnn2_dp03_vin_norm/model_w_best.pt')
    # debug_1 = torch.load('./model/3head_t_rnn_v_linear_cnn2_dp03_vin_norm/debug_1.pt')
    # debug_2 = torch.load('./model/3head_t_rnn_v_linear_cnn2_dp03_vin_norm/debug_2.pt')
    #
    # best_model = best_model['state_dict']
    # debug_1 = debug_1['state_dict']
    # debug_2 = debug_2['state_dict']
    #
    #
    # print(best_model['w2v.weight'][0,:10])
    # print(debug_1['w2v.weight'][0,:10])
    #
    # for key in best_model.keys():
    #     # print(type(best_model[key]))
    #     assert np.all(best_model[key].cpu().numpy() == debug_1[key].cpu().numpy())
    #     assert np.all(best_model[key].cpu().numpy() == debug_2[key].cpu().numpy())
    #     assert np.all(debug_1[key].cpu().numpy() == debug_2[key].cpu().numpy())


    result_1 = np.load('./model/3head_t_rnn_v_linear_cnn2_dp03_vin_norm/result_1.npz')
    result_2 = np.load('./model/3head_t_rnn_v_linear_cnn2_dp03_vin_norm/result_2.npz')
    assert np.all(result_1['e_vs'] == result_2['e_vs'])
    assert np.all(result_1['e_ts'] == result_2['e_ts'])
