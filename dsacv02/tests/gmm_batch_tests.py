from actor_critic_test import *

"""
Construct a GMM from critic parameters
"""
stds_cr.abs_()
start_cr_loop = time.perf_counter()
gmms_cr = []
for idx_cr, (mean_cr_i, std_cr_i, kernel_weight_cr_i) in enumerate(zip(means_cr, stds_cr, kernel_weights_cr)):
    gmm_i = generate_gmm_distr(mean=mean_cr_i.squeeze(), std=std_cr_i.squeeze(),
                               kweight=kernel_weight_cr_i.squeeze())
    gmms_cr.append(gmm_i)
    # print(f'Sample from Critic GMM {idx_cr}: {gmm_i.rsample()}')
end_cr_loop = time.perf_counter()
print(f'\n Calculation time for critic loop: {end_cr_loop - start_cr_loop}')
# print('----------------\n')
""" 
Construct a GMM from actor parameters
"""
# In-place operation, in addition, separate for actions
stds_act.abs_()
start_act_loop = time.perf_counter()
gmms_act = []
actions1_gmm = []
actions2_gmm = []
# Action 1 + Action 2 in same loop
for idx_act, (mean_act_i, std_act_i, kernel_weight_act_i) in enumerate(zip(means_act, stds_act, kernel_weights_act)):
    action1_i, action1_log_prob_i = actor.sample_from_action_distr(locs=mean_act_i[:, 0],
                                                                   stds=std_act_i[:, 0],
                                                                   k_weights=kernel_weight_act_i,
                                                                   reparameterization=True)
    action2_i, action2_log_prob_i = actor.sample_from_action_distr(locs=mean_act_i[:, 1],
                                                                   stds=std_act_i[:, 1],
                                                                   k_weights=kernel_weight_act_i,
                                                                   reparameterization=True)

    gmm_act1_i = generate_gmm_distr(mean=mean_act_i[:, 0], std=std_act_i[:, 0],
                                    kweight=kernel_weight_act_i)
    gmm_act2_i = generate_gmm_distr(mean=mean_act_i[:, 1], std=std_act_i[:, 1],
                                    kweight=kernel_weight_act_i)
    actions1_gmm.append(gmm_act1_i)
    actions2_gmm.append(gmm_act2_i)
    # print(f'Sample for action 2 from actor GMM {idx_act}: {action2_i} with log_prob {action2_log_prob_i}')
    # print(f'Sample for action 1 from actor GMM {idx_act}: {action1_i} with log_prob {action1_log_prob_i}')
gmms_act.append(actions1_gmm)
gmms_act.append(actions2_gmm)
end_act_loop = time.perf_counter()
print(f'Calculation time for actor loops over all actions: {end_act_loop - start_act_loop}')
'''
Check sampling with standard implementation and own implementation
'''
mean1_act1 = means_act[0][:, 0]
std1_act1 = stds_act[0][:, 0]
kweight1_act1 = kernel_weights_act[0]
check_sampling(mean=mean1_act1, std=std1_act1, kweight=kweight1_act1, n_sampling=10_000)