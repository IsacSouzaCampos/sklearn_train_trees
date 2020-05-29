import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def save_names_file(nfeat, filename):
	f = open(filename, 'w')
	print('Fout.\n', file=f)
	for i in range(nfeat):
		print('X%d:\t0,1.' % i, file=f)

	print('Fout:\t0,1.', file=f)
	f.close()


output_csv = open('c50_results.csv', 'w')
print(','.join(['base_name', 'sk_acc_tree', 'sk_acc_rf', 'tr_acc', 'te_acc', 'eq_one', 'eq_zero']), file = output_csv)

for path in os.listdir('../../redundancy_corrector/non_redundant_benchmarks/'):
	if '.train' in path:
		print(path)

print('='*20)

for path in os.listdir('../../redundancy_corrector/non_redundant_benchmarks/'):
	if '.train' not in path or '.txt' in path:
		continue
	print(path)
	base_name = path.split('.train')[0]
	c50f_data = base_name + '.data'
	c50f_names = base_name + '.names'
	c50f_test = base_name + '.test'
	c50f_output = base_name + '.out'

	ftrain = open('../../redundancy_corrector/non_redundant_benchmarks/'+path, 'r')
	ftest = open('../../redundancy_corrector/non_redundant_benchmarks/'+path.replace('.train', '.valid'), 'r')

	lines = ftrain.readlines()
	lines_test = ftest.readlines()
	ftrain.close()
	ftest.close()

	train_data = []
	test_data = []
	for line in lines:
		if '.' in line:
			continue
		x, y = line.split()
		x = [_ for _ in x]
		train_data.append(x+[y])

	for line in lines_test:
		if '.' in line:
			continue
		x, y = line.split()
		x = [_ for _ in x]
		test_data.append(x+[y])

	train_data = np.array(train_data)
	test_data = np.array(test_data)
	np.savetxt(c50f_data, train_data, fmt='%c', delimiter=',')
	np.savetxt(c50f_test, test_data, fmt='%c', delimiter=',')
	save_names_file(nfeat=train_data.shape[1]-1, filename=c50f_names)


output_csv.close()
