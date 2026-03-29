import os

# default_path = r"~/.openclaw/workspace/skills"
# whitelist = r"~/.openclaw/workspace/skills/skill-guard/whitelist.txt"
default_path = r"./"    # 测试环境
whitelist = r"./skill-guard/whitelist.txt"

def check_skill() -> list[str]:
    """查找已下载且不在白名单中的skills

    Returns:
        list[str]: 不在白名单中的skills列表
    """
    files = os.listdir(default_path)
    waited_for_check = []
    with open(whitelist, "r") as f:
        whitelist_files = f.read().splitlines()
    for file in files:
        if file not in whitelist_files:
            waited_for_check.append(file)
    return waited_for_check

def check_filefolder(prefix: str) -> list[str]:
    """递归查找文件夹

    Args:
        prefix (str): 文件夹前缀

    Returns:
        list[str]: 文件夹内容路径列表
    """
    filefolder = os.listdir(prefix)
    returns = []
    for file in filefolder:
        if os.path.isdir(os.path.join(prefix, file)):
            returns.extend(check_filefolder(os.path.join(prefix, file)))
        else:
            returns.append(os.path.join(prefix, file))
    return returns

def check_single_skill(skill_name: str) -> list[str]:
    """检查单个skill文件夹中的文件

    Args:
        skill_name (str): skill名称

    Returns:
        list[str]: skill文件夹中所有文件的路径列表
    """
    path = os.path.join(default_path, skill_name)
    return check_filefolder(path)

if __name__ == "__main__":
    print(os.getcwd())
    for skill in check_skill():
        print(f"Skill: {skill}")
        files = check_single_skill(skill)
        for file in files:
            print(f"  - {file}")