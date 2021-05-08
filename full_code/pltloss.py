from matplotlib import pyplot as plt
f = open('../logs/logs_avg_kl.txt', 'r')
lines = f.readlines()
i = 0
train = []
sum_tx = 0
for line in lines:
	a = line.split(' ')
	leng = len(a)
	x = a[leng - 1]
	strleng = len(x)
	tx = x[0:strleng - 1]
	#print(tx)
	train.append(float(tx))
	sum_tx += float(tx)
	i += 1
print('AVG', sum_tx / float(i))
f.close()
fig = plt.figure(dpi=200)

ax1 = plt.subplot(121)
plt.plot(train, label='train')
#ax1.set_ylim([0, 0.1])
plt.xlabel('Iters')
plt.ylabel('Y')
plt.title('Y')
plt.legend()
fig.savefig('train.png')        
