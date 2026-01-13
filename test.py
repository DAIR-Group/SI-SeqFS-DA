from modules import gendata, SI_SeqFS_DA, DA_SeqFS
import numpy as np
ns = 30
nt = 15
truebeta= [1, 0,  2, 0]
p = len(truebeta)
print('true beta:', truebeta)

true_beta_s = np.full((p,1), 2) #source's beta
true_beta_t = np.array(truebeta).reshape((-1,1)) #target's beta

K = 2 # select k features
# generate data
Xs, Xt, Ys, Yt, Sigma_s, Sigma_t = gendata.generate(ns, nt, p, true_beta_s, true_beta_t)

# apply DA-SeqFS to select features
list_selected_features = DA_SeqFS.DA_SeqFS(Xs, Ys, Xt, Yt, Sigma_s, Sigma_t,K, method='forward')

# SI-SeqFS-DA to calculate p-value of the selected features
print('Index of selected features:', '{' + ', '.join(map(str, list_selected_features)) + '}')
print('Applying SI-SeqFS-DA to calculate p-value of the selected features:')
for j in range(len(list_selected_features)):
    pvalue = SI_SeqFS_DA.SI_SeqFS_DA(Xs, Ys, Xt, Yt, K, Sigma_s, Sigma_t, method='forward', jth = j)
    print(f'    p-value of feature {list_selected_features[j]}: {pvalue}')