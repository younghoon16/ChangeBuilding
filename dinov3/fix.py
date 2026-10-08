# fix_torch_compiler.py
import os
import re


def fix_compiler_decorators(content):
    """修复torch.compiler装饰器"""
    if 'torch.compiler' not in content:
        return content, False

    lines = content.split('\n')
    new_lines = []
    fixed = False

    for line in lines:
        if '@torch.compiler.' in line and not line.strip().startswith('#'):
            # 检查是否是allow_in_graph
            if 'allow_in_graph' in line:
                # 用try-except包装
                new_lines.append('try:')
                new_lines.append(f'    {line}')
                new_lines.append('except AttributeError:')
                new_lines.append('    # PyTorch < 2.1, skip compiler decorator')
                new_lines.append('    pass')
                fixed = True
            elif 'disable' in line:
                new_lines.append('try:')
                new_lines.append(f'    {line}')
                new_lines.append('except AttributeError:')
                new_lines.append('    # PyTorch < 2.1')
                new_lines.append('    pass')
                fixed = True
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return '\n'.join(new_lines), fixed


def fix_all_files(directory='./'):
    """修复目录下的所有文件"""
    fixed_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)

                with open(filepath, 'r') as f:
                    content = f.read()

                if 'torch.compiler' in content:
                    fixed_content, changed = fix_compiler_decorators(content)

                    if changed:
                        with open(filepath, 'w') as f:
                            f.write(fixed_content)
                        fixed_count += 1
                        print(f"✓ Fixed: {filepath}")

    return fixed_count


if __name__ == '__main__':
    print("修复torch.compiler引用...")
    fixed = fix_all_files()
    print(f"\n修复了 {fixed} 个文件")