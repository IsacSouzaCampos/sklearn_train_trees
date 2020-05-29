import time
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from copy import deepcopy
import numpy as np
import itertools
import concurrent.futures


class TrainTrees:
    @staticmethod
    def pythonize_sop(sop):
        or_list = []
        expr_ = ''
        for ands in sop:
            and_list = []
            and_expr = '('
            for attr, negated in ands:
                if negated == 'true':
                    and_list.append('not(x%s)' % attr)
                else:
                    and_list.append('(x%s)' % attr)
            and_expr += ' and '.join(and_list)
            and_expr += ')'
            or_list.append(and_expr)
        expr_ += ' or '.join(or_list)

        return expr_

    @staticmethod
    def tree_to_sop(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

        ors = []

        def recurse(node, depth, expression):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]

                recurse(tree_.children_left[node], depth + 1, deepcopy(expression + [[name, 'true']]))

                recurse(tree_.children_right[node], depth + 1, deepcopy(expression + [[name, 'false']]))
            else:
                if np.argmax(tree_.value[node]) == 1:
                    ors.append(deepcopy(expression))

        recurse(0, 1, [])

        return ors

    @staticmethod
    def train_tree(_max_depth, train_data, test_data):
        xtr, ytr = train_data[:, :-1], train_data[:, -1]
        xte, yte = test_data[:, :-1], test_data[:, -1]

        tree = DecisionTreeClassifier(max_depth=_max_depth)
        tree.fit(xtr, ytr)

        ypred_tree = tree.predict(xte)
        acc_tree = (ypred_tree == yte).mean()

        return tree, acc_tree

    @staticmethod
    def get_number_of_inputs(_path):
        with open(f'IWLS2020-benchmarks/{_path}') as file:
            for line in file:
                if '.i' in line:
                    return int(line.split(' ')[1])

    def execute_methods(self, _dir_path, _path, _max_depth):
        base_name = _path.split('.data')[0]
        c50f_data = base_name + '.data'
        c50f_test = base_name + '.test'

        _train_data = np.loadtxt(f'{_dir_path}/{c50f_data}', dtype='int', delimiter=',')
        test_data = np.loadtxt(f'{_dir_path}/{c50f_test}', dtype='int', delimiter=',')
        feature_names = list(map(str, list(range(_train_data.shape[1]))))
        tree, acc_tree = self.train_tree(_max_depth, _train_data, test_data)
        sop_tree = self.tree_to_sop(tree, feature_names)

        expr = self.pythonize_sop(sop_tree)

        output_str = f'{base_name} {acc_tree * 100}\n'
        print(f'{base_name} {acc_tree * 100}')
        total_acc_tr = acc_tree * 100

        return output_str, total_acc_tr, expr

    @staticmethod
    def order_ex_results(string):
        dictionary = {}
        ex_list = []
        lines = string.splitlines()
        for line in lines:
            item = line.split()
            dictionary[int(item[0].replace('ex', ''))] = item[1]
            ex_list.append(int(item[0].replace('ex', '')))

        result = ''
        for i in sorted(ex_list):
            result += f'{dictionary[i]}\n'

        return result

    @staticmethod
    def aig_maker(_dir_path):
        file = open('mltest.txt', 'w+')
        file.truncate(0)
        file.close()

        for _path in os.listdir(_dir_path):
            if '.train' in _path:
                new_file = str(f'mix_train_valid/aig/{_path[:4]}.train.aig')
                script = str(f'read_pla {_dir_path}/{_path}\nstrash\nwrite_aiger {new_file}\n&read {new_file}; &ps;'
                             f'&mltest mix_train_valid/valid/{_path[:4]}.valid.pla')

                script_file = open('script.scr', 'w+')
                script_file.write(script)
                script_file.close()

                os.system('./abc -c \'source script.scr\' >> mltest.txt')
                print(f'{_path[:4]}.train.aig finished')

    def mltest_data_maker(self):
        num_of_ands = ''
        accuracy = ''

        with open('mltest.txt') as file:
            ex = ''
            for line in file:
                if 'Total' in line:
                    vec = line.split('  ( ')
                    accuracy += f'{ex} {vec[1][:5]}\n'
                elif 'and = ' in line:
                    i = line.find('ex')
                    ex = line[i:i + 4]
                    vec = line.split(' = ')
                    int_num = ''
                    for num in vec[2].replace('lev', '').replace(' ', ''):
                        if num.isalnum():
                            int_num += num
                    int_num = int_num.replace('0m135m', '')
                    num_of_ands += f'{ex} {int_num}\n'
                elif 'does not match the AIG' in line:
                    accuracy += f'{ex} failed\n'

        num_of_ands_file = open('mix_train_valid/ands.txt', 'w+')
        accuracy_file = open('mix_train_valid/accuracy.txt', 'w+')
        print(num_of_ands)
        num_of_ands_file.write(self.order_ex_results(num_of_ands))
        print(accuracy)
        accuracy_file.write(self.order_ex_results(accuracy))
        num_of_ands_file.close()
        accuracy_file.close()

    def mix_train_valid(self, _dir_path, _path):
        _path = _path.replace('.data', '.train.pla')
        header = ''
        body = ''
        with open(f'{_dir_path}/{_path}') as file:
            for line in file.read().splitlines():
                if '.' in line:
                    header += f'{line}\n'
                else:
                    body += f'{line}\n'
        _path = _path.replace('train', 'valid')
        with open(f'{_dir_path}/{_path}') as file:
            for line in file.read().splitlines():
                if '.' in line:
                    continue
                body += f'{line}\n'

        # body = self.correct_redundancy(body)
        _train_data, _valid_data = self.split_train_valid(header, body)
        return _train_data, _valid_data, _path[:4]

    @staticmethod
    def split_train_valid(header, body):
        train_body = ''
        valid_body = ''
        c = 0
        for line in body.splitlines():
            if c % 4:
                train_body += f'{line}\n'
            else:
                valid_body += f'{line}\n'
            c += 1

        train_header = ''
        valid_header = ''
        for line in header.splitlines():
            if '.e' in line:
                continue
            elif '.p' in line:
                train_header += f'.p {len(train_body.splitlines())}\n'
                valid_header += f'.p {len(valid_body.splitlines())}\n'
            else:
                train_header += f'{line}\n'
                valid_header += f'{line}\n'

        _train_data = f'{train_header}{train_body}.e'
        _valid_data = f'{valid_header}{valid_body}.e'

        return _train_data, _valid_data

    @staticmethod
    def correct_redundancy(body):
        lines_dict = {}
        lines = body.splitlines()
        for line in lines:
            vec = line.split(' ')
            lines_dict[vec[0]] = vec[1]
        return '%s' % '\n'.join([f'{key} {value}' for (key, value) in lines_dict.items()])


if __name__ == '__main__':
    start = time.perf_counter()

    dir_path = 'IWLS2020-benchmarks'
    tests = [15]
    acc_tree_means = []
    acc_tree_mean_dict = {}

    # TrainTrees().aig_maker(dir_path)
    # TrainTrees().mltest_data_maker()

    for test in tests:
        output_string = ''
        print('base_name acc_tree')
        total_acc_tree = 0

        print('*' * 20)
        print(f'_max_depth = {test}')
        max_depth = test

        results = []
        count = 0
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     for path in os.listdir(dir_path):
        #         if '.train' not in path:
        #             continue
        #         # if count == 1:
        #         #     break
        #         count += 1
        #
        #         results.append(executor.submit(TrainTrees().mix_train_valid, dir_path, path))
        #
        #     for r in results:
        #         train_data, valid_data, ex = r.result()
        #
        #         train_output = open(f'mix_train_valid/train/{ex}.train.pla', 'w+')
        #         valid_output = open(f'mix_train_valid/valid/{ex}.valid.pla', 'w+')
        #         train_output.write(train_data)
        #         valid_output.write(valid_data)
        #         train_output.close()
        #         valid_output.close()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for path in os.listdir(dir_path):
                if '.train' not in path:
                    continue
                # if count == 1:
                #     break
                count += 1

                results.append(executor.submit(TrainTrees().execute_methods, dir_path, path, 15))

            for r in results:
                out_str, tot_ac_tr, exp = r.result()

                expr_output = open('')
