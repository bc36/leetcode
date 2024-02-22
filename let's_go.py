import html, logging, json, os, re, subprocess, sys
from typing import Callable, Dict, List, Tuple

"""
功能:
根据输入题目网址, 获取题目信息, 生成代码模版, 测试模版及测试数据

如何使用:
> python3 let\'s_go.py https://leetcode.com/problems/add-two-numbers/
> python3 let\'s_go.py https://leetcode.com/problems/add-two-integers/description/
或者
> python3 let\'s_go.py
> 后续再传入 URL


https://leetcode.com/problems/two-sum/ 输出任意符合结果的答案, 测试结果不稳定
https://leetcode.com/problems/add-two-numbers/ 树, 节点
https://leetcode.com/problems/lru-cache/ 设计题
https://leetcode.com/problems/course-schedule/ 图
https://leetcode.com/problems/count-the-number-of-powerful-integers/


Other related things:
curl 'https://leetcode.com/contest/api/info/weekly-contest-379/' | jq
"""


WHOAMI = subprocess.run(["whoami"], capture_output=True, text=True).stdout
DIR = f"/Users/{WHOAMI.strip()}/workspace/leetcode/Go"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s: %(message)s",
)


"""
[a-zA-Z0-9\-/]: 匹配任意大小写字母、数字、短横线（减号）或斜杠
+: 匹配前面的表达式（即[a-zA-Z0-9\-/]）一次或多次
/?: 匹配零次或一次斜杠
$: 匹配输入的结尾
"""
valid_pattern: Callable[[str], str] = lambda url: re.compile(
    r"https://leetcode\.com/problems/[a-zA-Z0-9\-/]+/?$"
).match(url)


def get_url() -> str:
    """
    输入 .com 的网址, .cn 无法获得代码片段

    """

    # 从命令行得到 URL 或者 后续再输入
    ret = None
    if len(sys.argv) == 2:
        ret = sys.argv[1]
    else:
        ret = input(
            "Pls enter a .com URL like: 'https://leetcode.com/problems/two-sum/': "
        )

    ret = ret.replace("/leetcode.cn/", "/leetcode.com/")
    ret = ret.replace("/description", "")  # /description 也可以获取内容, 似乎不影响结果
    ret = ret.strip()  # Removeleading and trailing whitespace
    if ret[-1] != "/":
        ret += "/"

    if not valid_pattern(ret):
        logging.error("Wrong URL pattern! ¯\_(ツ)_/¯")
        sys.exit(1)

    return ret


def get_info(url: str) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    原始命令
    curl -s "https://leetcode.com/problems/count-the-number-of-powerful-integers/" | grep -o '<script id[^<]*<\/script>' | gsed 's/^.\{51\}//;s/.\{9\}$//' | jq
    """

    cmd = f"curl -s \"{url}\" | grep -o '<script id[^<]*<\\/script>' | gsed 's/^.\{{51\}}//;s/.\{{9\}}$//' | jq"
    output = subprocess.check_output(cmd, shell=True, text=True)  # 全部信息
    obj = json.loads(output)  # 将 JSON 字符串解析为 Python 对象
    try:
        """
        queries[0]:  question meta data 题目基础信息
        queries[1]:  languageList
        queries[6]:  hints
        queries[7]:  content 题目描述, 题意
        queries[8]:  stat 统计数据, totalAccepted totalSubmission totalAcceptedRaw totalSubmissionRaw acRate
        queries[9]:  topicTags 题目归类
        queries[11]: codeSnippets 各语言板子
        """
        queries = obj["props"]["pageProps"]["dehydratedState"]["queries"]
        meta_data, hints, content, topic_tags, code_snippets = (
            queries[0],
            queries[6],
            queries[7],
            queries[9],
            queries[11],
        )
        # Go 不固定在哪个下标, 所以扫一遍
        for section in code_snippets["state"]["data"]["question"]["codeSnippets"]:
            if section["lang"] == "Go":
                code_snippet = section
    except KeyError as e:
        logging.error(f"Error: {e}. One or more nested keys not found. ¯\_(ツ)_/¯\n")
        sys.exit(1)  # comment this line for more info

    return meta_data, hints, content, topic_tags, code_snippet["code"]


def extract_meta_data(meta_data: Dict) -> Tuple[str, str, str, str]:
    """
    questionId maybe different from questionFrontendId
    """

    question = meta_data["state"]["data"]["question"]
    return (
        question["questionFrontendId"],
        question["title"],
        question["titleSlug"],
        question["difficulty"],
    )


def remove_html_tags(text: str) -> str:
    """
    'context' is in HTML format.
    'mysqlSchemas' and 'dataSchemas' included here
    """

    # 去除 <p></p>, <strong></strong>, <pre></pre> 等 HTML 标签
    clean = re.compile("<.*?>")

    # 转换 &nbsp; 空格 / &quot; 左双引号 / &quot; 右双引号 之类的东西
    encoded_text = re.sub(clean, "", text)
    decoded_text = html.unescape(encoded_text)
    return decoded_text


def parse_code(code: str) -> Tuple[str, bool, List[int]]:
    """
    从 code 中, 解析函数名, 题目也可能是构造题
    TODO: func_los 可能没用

    细分的话有四种题目类型:
    函数-无预定义类型, 绝大多数题目都是这个类型
    函数-有预定义类型, 如 LC174C
    方法-无预定义类型, 如 LC175C
    方法-有预定义类型, 如 LC163B
    """

    lines = code.split("\n")

    if "func Constructor(" in code:
        func_los = []
        for i, line in enumerate(lines):
            if line.startswith("func Constructor("):  # 构造器定义
                func_los.append(i)
            elif line.startswith("func ("):  # 方法定义
                func_los.append(i)
        func_name, is_func_problem = "Constructor", False
    else:
        for i, line in enumerate(lines):
            if line.startswith("func "):  # 函数定义
                j = line.index("(")
                func_name, is_func_problem, func_los = line[5:j], True, [i]
                break
    return func_name, is_func_problem, func_los


def create_problem_file(
    dir_path: str, url: str, title_slug: str, difficulty: str, code: str
) -> None:
    problem_file_path = f"{dir_path}/{title_slug}.go"

    with open(problem_file_path, "w") as file:
        # TODO: 增加所属周赛 URL
        extra_import = '\n//lint:ignore ST1001 ¯\_(ツ)_/¯\nimport . "github.com/bc36/leetcode/Go/testutils"\n'
        comment_ending = code.index("*/") + 2 if "*/" in code else 0
        leading_new_line = "\n" if comment_ending == 0 else ""

        multiline = f"""package main
{extra_import if 'Definition' in code else ''}
// {difficulty}
// {url.replace('com', 'cn')}
// {url}
{leading_new_line}{code[comment_ending:]}
"""
        file.write(multiline)
    return


def create_test_file(
    dir_path: str, title_slug: str, func_name: str, is_func_problem: bool
) -> None:
    test_file_path = f"{dir_path}/{title_slug}_test.go"

    test_util_func, extra_log_info = "testutils.RunLeetCodeFuncWithFile", ""
    if not is_func_problem:
        test_util_func, extra_log_info = (
            "testutils.RunLeetCodeClassWithFile",
            "\n\t" + 't.Log("记得初始化所有全局变量")',
        )

    dir_path = '"+os.Getenv("USER")+"'.join(dir_path.split(os.environ.get("USER")))

    with open(test_file_path, "w") as file:
        multiline = f"""package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_{title_slug}(t *testing.T) {{{extra_log_info}
	targetCaseNum := 0 // -1
	if err := {test_util_func}(t, {func_name}, "{dir_path}/{title_slug}_test_cases.txt", targetCaseNum); err != nil {{
		t.Fatal(err)
	}}
}}
"""
        file.write(multiline)
    return


def create_data_file(dir_path: str, title_slug: str, content: str) -> None:
    data_file_path = f"{dir_path}/{title_slug}_test_cases.txt"
    lines = content.split("\n")
    multiline = ""
    for i, line in enumerate(lines):
        if line.startswith("Input: "):
            multiline += (
                "\n".join(equa[equa.index("=") + 2 :] for equa in line.split(", "))
                + "\n"
            )
        elif line.startswith("Output: "):
            multiline += line[line.index(":") + 2 :] + "\n\n"
        elif line == "Input":  # LC146
            while 1:
                i += 1
                if not lines[i].startswith("Output"):
                    multiline += lines[i] + "\n"
                else:
                    break
        elif line == "Output":
            multiline += lines[i + 1] + "\n\n"

    with open(data_file_path, "w") as file:
        file.write(multiline[:-1])
    return


if __name__ == "__main__":
    logging.info("'gsed' is required if using MacOS. Don't use 'leetcode.cn'")

    input_url = get_url()
    logging.info(f"The URL is '{input_url}'")

    """
    1. Preprare information
    """
    meta_data, hints, content, topic_tags, code_snippet = get_info(input_url)
    content = remove_html_tags(content["state"]["data"]["question"]["content"])
    question_frontend_id, title, title_slug, difficulty = extract_meta_data(meta_data)
    func_name, is_func_problem, func_los = parse_code(code_snippet)
    # hints: List[str] = hints["state"]["data"]["question"]["hints"]
    # topic_tags: List[Dict[str, str]] = topic_tags["state"]["data"]["question"]["topicTags"]
    logging.info(f"Fetched: {question_frontend_id} - {title} - {difficulty}")

    """
    2. Generate Go template
    """
    question_frontend_id = "{:04d}".format(int(question_frontend_id))
    title_slug = title_slug.replace("-", "_", -1)

    # Check if the directory exists
    dir_path = os.path.join(DIR, f"{DIR}/{question_frontend_id}")
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        logging.error("Directory already exists. ¯\_(ツ)_/¯")
        sys.exit(1)

    os.makedirs(dir_path)
    create_problem_file(dir_path, input_url, title_slug, difficulty, code_snippet)
    create_test_file(dir_path, title_slug, func_name, is_func_problem)
    create_data_file(dir_path, title_slug, content)
    logging.info(
        "The problem file, test file and test cases file are created. Let's go. :)"
    )
