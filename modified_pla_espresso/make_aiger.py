import os


source_path = 'Benchmarks_2_espresso'
target_path = 'Benchmarks_2_espresso_aig/original'

for file_name in os.listdir(source_path):
    if '.pla' in file_name:
        new_file = str(f'{target_path}/{file_name[:4]}.aig')
        script = str(f'read_pla {source_path}/{file_name}\nstrash\nwrite_aiger {new_file}')

        script_file = open('script.scr', 'w+')
        script_file.write(script)
        script_file.close()

        os.system('.././abc -c "source script.scr"')
        print(f'{file_name[:4]}.aig finished')

# output = ''
# for file_name in os.listdir(target_path):
#     output += f'{file_name[2:4]} '
# print(', '.join(output.split(' ')))
