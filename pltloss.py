from matplotlib import pyplot as plt
f = open('logs/logs_loss_total.txt', 'r')
lines = f.readlines()
i = 0
train = []
for line in lines:
	a = line.split(' ')
	leng = len(a)
	x = a[leng - 1]
	strleng = len(x)
	tx = x[0:strleng - 1]
	#print(tx)
	train.append(float(tx))
	
	i += 1
	
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
