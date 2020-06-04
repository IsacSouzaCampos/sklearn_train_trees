import os
from pyeda.inter import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from copy import deepcopy
import numpy as np
import concurrent.futures


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


def optimize_sop(_dir_path, _path):
    with open(f'{_dir_path}/{_path}') as file:
        string = file.read()
        inorder = string.split(';')[0].replace('INORDER = ', '').split(' ')
        inorder_vars = list(map(exprvar, inorder))
        values_dict = {}
        for i in range(len(inorder)):
            values_dict[inorder[i]] = inorder_vars[i]

        _expr = string.split('z1 = ')[1].replace(' ', '').replace('(', '').replace(')', '')

        or_splitted = _expr.split('+')
        and_exprs_lst = []
        for _ in or_splitted:
            and_logic = _.split('*')
            and_expr = 'And('
            for element in and_logic:
                if '!' in element:
                    and_expr += f'~vars_dict["{element.replace("!", "")}"], '
                else:
                    and_expr += f'vars_dict["{element}"], '
            and_expr += ')'
            and_exprs_lst.append(and_expr)
        final_expr = 'Or('
        final_expr += ', '.join([element for element in and_exprs_lst])
        final_expr += ')'

        change_optimize_sop_file(inorder, final_expr)
        os.system('python3 optimize_sop.py')
        remake_eqn(f'mix_train_valid/trained_trees_sop/{_path}', 'simplified_expr.txt')


def remake_eqn(full_path, simplified_string):
    _expr = ''
    _new_expr = ''
    with open(simplified_string) as file:
        for line in file.read().splitlines():
            _expr += line
    _and_exprs = _expr.replace('Or(And', '')[:-1].split(', And')
    _new_expr = ' + '.join(element.replace(', ', ' * ') for element in _and_exprs)
    _new_expr = _new_expr.replace('~', '!').replace('x', '(x').replace(' *', ') *').replace(' +', ') +')

    print(_new_expr)
    final_eqn = ''
    with open(full_path) as file:
        for line in file.read().splitlines():
            final_eqn += f'z1 = {_new_expr}' if 'z1 =' in line else f'{line}\n'
    final_eqn = f'{final_eqn.replace(" * )", ")")});'

    file = open(f'mix_train_valid/optimized_sop/{full_path[-8::]}', 'w+')
    file.write(final_eqn)
    file.close()


def change_optimize_sop_file(inorder, _expr):
    space = '    '
    with open('optimize_sop.py') as file:
        string = file.read()
        new_string = ''

        for line in string.splitlines():
            if 'continue' in line:
                continue
            elif 'inorder =' in line:
                new_string += f'inorder = ['
                new_string += ', '.join(f'"{element}"' for element in inorder)
                new_string += ']\n\n'
                new_string += f'inorder_vars = list(map(exprvar, inorder)){space}# continue\n'
                new_string += f'vars_dict = {"{}"}{space*8}# continue\n'
                new_string += f'for i in range(len(inorder)):{space*4}# continue\n'
                new_string += f'{space}vars_dict[inorder[i]] = inorder_vars[i]{space}# continue'
            elif 'optimized_eqn =' in line:
                new_string += f'{space}optimized_eqn = {_expr.replace(", )", ")").replace(";", "")}\n'
            else:
                new_string += f'{line}\n'

    file = open('optimize_sop.py', 'w+')
    file.write(new_string)
    file.close()


def train_tree(_max_depth, train_data, test_data):
    xtr, ytr = train_data[:, :-1], train_data[:, -1]
    xte, yte = test_data[:, :-1], test_data[:, -1]

    tree = DecisionTreeClassifier(max_depth=_max_depth)
    tree.fit(xtr, ytr)

    ypred_tree = tree.predict(xte)
    acc_tree = (ypred_tree == yte).mean()

    return tree, acc_tree


def get_number_of_inputs(_path):
    with open(_path) as file:
        for line in file:
            if '.i' in line:
                return int(line.split(' ')[1])


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


def aig_maker(_path_dir, _target_dir):   # dir_path = path to eqn files
    file = open('mltest.txt', 'w+')
    file.truncate(0)
    file.close()

    for _path in os.listdir(_path_dir):
        if '.eqn' in _path:
            new_file = str(f'{_target_dir}/{_path[:4]}.train.aig')
            script = str(f'read_eqn {_path_dir}/{_path}\nstrash\nwrite_aiger {new_file}\n&read {new_file}; &ps;'
                         f'&mltest mix_train_valid/benchmarks/{_path[:4]}.valid.pla')

            script_file = open('script.scr', 'w+')
            script_file.write(script)
            script_file.close()

            os.system('./abc -c \'source script.scr\' >> mltest.txt')
            print(f'{_path[:4]}.train.aig finished')


def mltest_data_maker(_results_target_dir):
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

    num_of_ands_file = open(_results_target_dir, 'w+')
    accuracy_file = open(_results_target_dir, 'w+')
    print(num_of_ands)
    num_of_ands_file.write(order_ex_results(num_of_ands))
    print(accuracy)
    accuracy_file.write(order_ex_results(accuracy))
    num_of_ands_file.close()
    accuracy_file.close()


def mix_train_valid(_dir_path, _path):
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
    _train_data, _valid_data = split_train_valid(header, body)
    return _train_data, _valid_data, _path[:4]


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


def correct_redundancy(body):
    lines_dict = {}
    lines = body.splitlines()
    for line in lines:
        vec = line.split(' ')
        lines_dict[vec[0]] = vec[1]
    return '%s' % '\n'.join([f'{key} {value}' for (key, value) in lines_dict.items()])


def eqn_maker(_expr, n_inputs):
    header = 'INORDER ='

    for i in range(n_inputs):
        header += f' x{i}'
    header += ';\nOUTORDER = z1;\n'

    body = 'z1 = '
    body += _expr.replace('not', '!').replace('and', '*').replace('or', '+')

    return f'{header}{body}'


def run(_dir_path, _path, _max_depth):
    base_name = _path.split('.data')[0]
    c50f_data = base_name + '.data'
    c50f_test = base_name + '.test'

    _train_data = np.loadtxt(f'{_dir_path}/{c50f_data}', dtype='int', delimiter=',')
    test_data = np.loadtxt(f'{_dir_path}/{c50f_test}', dtype='int', delimiter=',')
    feature_names = list(map(str, list(range(_train_data.shape[1]))))
    tree, acc_tree = train_tree(_max_depth, _train_data, test_data)
    sop_tree = tree_to_sop(tree, feature_names)

    _expr = pythonize_sop(sop_tree)

    output_str = f'{base_name} {acc_tree * 100}\n'
    # print(f'{base_name} {acc_tree * 100}')
    total_acc_tr = acc_tree * 100

    return output_str, total_acc_tr, _expr

# para minimizar sop's, olhar:
# https://pyeda.readthedocs.io/en/latest/2llm.html


acc_tree_means = []
acc_tree_mean_dict = {}

results = []
count = 0

dir_path = 'Benchmarks_2'
target_path = 'Benchmarks_2_aig'
for path in os.listdir(dir_path):
    if '.train' not in path and '.valid' not in path:
        continue
    command = str(f'read_pla {dir_path}/{path}\nstrash\nwrite_aiger {target_path}/'
                  f'{path.replace(".pla", ".aig")}')

    script = open('script.scr', 'w+')
    script.write(command)
    script.close()

    os.system('./abc -c "source script.scr"')

# for path in os.listdir('mix_train_valid/trained_trees_sop'):
#     if '.eqn' not in path:
#         continue
#     # if count == 1:
#     #     break
#     #     count += 1
#     print(path)
#     optimize_sop('mix_train_valid/trained_trees_sop', path)

# with concurrent.futures.ProcessPoolExecutor() as executor:
#     for path in os.listdir(dir_path):
#         if '.eqn' not in path:
#             continue
#         # if count == 1:
#         #     break
#         count += 1
#
#         results.append(executor.submit(optimize_sop, dir_path, path))
#
#         for r in results:
#             out_str, tot_ac_tr, exp = r.result()
#             exp = exp.replace('not', '!').replace('and', '*').replace('or', '+')
#
#             expr_output = open(f'mix_train_valid/train_trees_sop/{out_str[:4]}.eqn', 'w+')
#             expr_output.write(eqn_maker(exp, get_number_of_inputs(f'mix_train_valid/benchmarks/'
#                                                                   f'{out_str[:4]}.train.pla')))
#             expr_output.close()
