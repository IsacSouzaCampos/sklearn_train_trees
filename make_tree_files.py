import numpy as np
import os


def save_names_file(nfeat, filename):
	f = open(filename, 'w')
	print('Fout.\n', file=f)
	for i in range(nfeat):
		print('X%d:\t0,1.' % i, file=f)

	print('Fout:\t0,1.', file=f)
	f.close()


dir_path = 'full_adder_example'

output_csv = open('c50_results.csv', 'w')
print(','.join(['base_name', 'sk_acc_tree', 'sk_acc_rf', 'tr_acc', 'te_acc', 'eq_one', 'eq_zero']), file=output_csv)

for path in os.listdir(dir_path):
	if '.train' in path and '.aig' not in path:
		print(path)

print('='*20)

for path in os.listdir(dir_path):
	if '.train' not in path or '.aig' in path:
		continue

	base_name = path.split('.train')[0]
	c50f_data = base_name + '.data'
	c50f_names = base_name + '.names'
	c50f_test = base_name + '.test'
	c50f_output = base_name + '.out'

	ftrain = open(f'{dir_path}/{path}', 'r')
	path = path.replace('.train', '.valid')
	ftest = open(f'{dir_path}/{path}', 'r')

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
	np.savetxt(f'full_adder_example/{c50f_data}', train_data, fmt='%c', delimiter=',')
	np.savetxt(f'full_adder_example/{c50f_test}', test_data, fmt='%c', delimiter=',')
	save_names_file(nfeat=train_data.shape[1]-1, filename=f'full_adder_example/{c50f_names}')


output_csv.close()
