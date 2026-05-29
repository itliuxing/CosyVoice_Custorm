import os


def get_project_root():
    """向上递归查找包含 reqs.txt 和 README.md 的目录"""
    current_dir = os.path.abspath(os.getcwd())  # 从当前工作目录开始

    while True:
        reqs_file = os.path.join(current_dir, "reqs.txt")
        readme_file = os.path.join(current_dir, "README.md")

        if os.path.exists(reqs_file) or os.path.exists(readme_file):
            return current_dir  # 找到匹配的项目根目录

        parent_dir = os.path.dirname(current_dir)  # 获取上一级目录
        if parent_dir == current_dir:  # 已经到达磁盘根目录
            return None

        current_dir = parent_dir  # 继续向上查找


def get_parent_dir(current_dir):
    """
    todo 找到当前目录的上级目录
    :param current_dir:
    """
    parent_dir = os.path.dirname(current_dir)  # 获取上一级目录
    return parent_dir

if __name__ == "__main__":
    print(get_project_root())  # 运行看看效果
