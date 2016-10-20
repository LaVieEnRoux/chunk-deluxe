# import subprocess 
# import numpy as np

# best_alpha = 0
# best_beta = 0
# best_gamma = 0
# best_kappa = 0
# best_acc   = 0
# for alpha in range(37700, 37900, 50):
# 	for beta in np.arange(0.07, 0.09, 0.01): 
# 		for gamma in np.arange(8.9, 9.0, 0.01):
# 			for kappa in np.arange(0.49, 0.51, 0.01):
# 				out = subprocess.check_output("python segment.py -a {0} -g {1} -k {2} -l {3} | python score-segments.py".format(alpha, gamma, kappa, beta), shell=True)
# 				acc = float(out.split()[1])
# 				if acc > best_acc:
# 					best_acc = acc
# 					best_alpha = alpha
# 					best_beta = beta
# 					best_gamma = gamma
# 					best_kappa = kappa
# 					print "Best so far:" 
# 					print alpha, beta, gamma, kappa, acc
# print "*******************************"
# print best_acc
# print best_alpha
# print best_beta
# print best_gamma
# print best_kappa

# #37800 0.071 8.98 0.45 91.24

amount = float(input("How much money do you have?"))
F = amount//5
print(F)
T = (amount-F*5)//2
print(T)
Q = (amount-F*5-T*2)//0.25
if ((amount-F*5-T*2-Q*0.25))>0.135:
    print(Q+1)
else:
    print(Q)