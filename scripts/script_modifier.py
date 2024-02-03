import os
import argparse

def modify(file_path, target, new_value):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            if f"{target}=" in line and '&' not in line:
                line = f"{target}={new_value}\n"
            file.write(line)

def modify_augmentation(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    with open(file_path, 'w') as file:
        for line in lines:
            if f"for s4rl_augmentation_type" in line:
                line = f"for s4rl_augmentation_type in 'identical'\n"
            file.write(line)

def modify_all_scripts_in_directory(root_dir, target, new_value):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.sh'):
                file_path = os.path.join(subdir, file)
                modify(file_path, target, new_value)
                print(f"Modified {target} in {file_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", dest='path', action='store', default="scripts/DT_GTA")
    parser.add_argument('-t', '--target', dest='target', action='store', default='GDA')
    parser.add_argument('-n', '--new_value', dest='new_value', action='store', default='GTA_1M_FINAL')

    args = parser.parse_args()

    path = args.path
    target = args.target
    new_value = args.new_value
    if target != 'augmentation':
        modify_all_scripts_in_directory(path, target, new_value)
    else:
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.sh'):
                    file_path = os.path.join(subdir, file)
                    modify_augmentation(file_path)
                    print(f"Modified {target} in {file_path}")

if __name__ == '__main__':
    main()