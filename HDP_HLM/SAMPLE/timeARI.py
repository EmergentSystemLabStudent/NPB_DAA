import matplotlib.pyplot as plt
import numpy as np

rilAve=[]
risAve=[]

for i in range(1,8):
	ril=np.loadtxt("SyntheticDemo_result_"+str(i)+"/summary_figs/aRIs_l.txt")
	ris=np.loadtxt("SyntheticDemo_result_"+str(i)+"/summary_figs/aRIs_s.txt")
	rilAve.append(ril)
	risAve.append(ris)
rilAve=np.array(rilAve)
risAve=np.array(risAve)

fig=plt.figure(figsize=(18,12))
ax=fig.add_subplot(111)
plt.xlim([1,100])
plt.ylim([0,1])

plt.xticks(fontsize=36)
plt.yticks(fontsize=36)
plt.ylabel("adjusted rand index",fontsize=48)
plt.xlabel("iteration",fontsize=48)

a=plt.errorbar(range(1,101),[np.average(rilAve[:,key]) for key in range(100)],yerr=[np.std(rilAve[:,key]) for key in range(100)])
b=plt.errorbar(range(1,101),[np.average(risAve[:,key]) for key in range(100)],yerr=[np.std(risAve[:,key]) for key in range(100)])
plt.legend(('letter','word'),fontsize=48,loc=4)
#plt.show()
plt.savefig("ARITIME.eps")
