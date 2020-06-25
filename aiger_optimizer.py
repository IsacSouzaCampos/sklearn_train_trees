import os


source_path = 'modified_pla_espresso/Benchmarks_2_espresso_aig/optimized'
target_path = 'modified_pla_espresso/Benchmarks_2_espresso_aig/optimized'

file = open(f'{target_path.split("/")[0]}/mltest.txt', 'w+')
file.truncate(0)
file.close()

for file_name in os.listdir(source_path):
    script = f'read_aiger {source_path}/{file_name}\nrefactor\nrewrite\nwrite_aiger {target_path}/{file_name}\n' \
             f'&read {target_path}/{file_name}; &ps; &mltest IWLS2020-benchmarks/' \
             f'{file_name.replace(".aig", ".valid.pla")}'
    file = open('script.scr', 'w+')
    file.write(script)
    file.close()

    os.system(f'./abc -c "source script.scr" >> {target_path.split("/")[0]}/mltest.txt')
