import os
from openai import OpenAI
from tqdm import tqdm
from jacksung.utils.time import Stopwatch, get_time_str


def get_en_polish_prompt(text):
    polish_prompt = \
        fr'''
        # Rewrite the text in an academic writing style, using more appropriate vocabulary and sentence structure while keeping the original meaning unchanged:
        - Make sure the rewritten version conveys the same information and intention as the original text.
        - Please output the rewritten text directly in latex format, without including the original text, thinking logic, comments, explanations, etc.
        - Do not output any control commands that do not exist in the input content (such as \documentclass, \begin, \end, etc.), just use latex format to output mathematical formulas, symbols, references, or other latex instructions contained in the input content.
        - Note that special symbols and formulas are output in latex format, not directly output special characters.
        - Only when the input content contains control codes such as \par, the code needs to be added to the corresponding position of the output content.
        - If the input content only contains code and does not contain any substantial text content, the input content is directly output without any changes. Be careful not to miss symbols such as brackets.
        - Make sure that the output content can be compiled normally after replacing the input content in the original document.
        The following is the input content:
        {text}
        '''
    return polish_prompt


def get_cn_polish_prompt(text):
    polish_prompt = \
        fr'''
        # 用学术写作风格重写下面的文本,在保持原本涵义不变的情况下使用更合适的词汇和句子结构:
        - 确保改写后的版本传达的信息和意图与原文相同。
        - 请直接以latex格式输出重写后的文本，不需要包含原文、思考逻辑、注释、解释说明等其他内容。
        - 不需要输出任何输入内容中不存在的控制命令（如\documentclass、\begin、\end等），只需要使用latex格式输出数学公式、符号、引用或者输入内容中所包含的其他latex指令。
        - 注意特殊符号和公式以latex格式输出，而不是直接输出特殊字符。
        - 仅在输入内容中包含\par类似的控制性代码时，在输出内容对应位置需要添加该代码。
        - 如果输入内容仅包含代码，不包含任何实质性的文本内容，则直接将输入内容不做任何改动输出。注意不要遗漏括号等符号。
        - 确保输出内容在原文档中替换输入内容后能够正常编译通过。
        - 重要：你需要确定输入内容所使用的语言，然后使用相同语言进行输出，以保证输入和输出语言一致。
        以下为输入内容:
        {text}
        '''
    return polish_prompt


def merge_content(tex_dir, main_tex):
    result_tex = ''
    with open(os.path.join(tex_dir, main_tex), 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith(r'\input{') or line.startswith(r'\include{'):
                sub_tex_path = os.path.join(tex_dir, line.split('{')[1].split('}')[0])
                if not sub_tex_path.endswith('.tex'):
                    sub_tex_path += r'.tex'
                result_tex += merge_content(tex_dir, sub_tex_path) + '\n'
            else:
                result_tex += line
    return result_tex


class AI:
    def __init__(self, token, base_url, model_name='deepseek-r1:70b'):
        self.client = OpenAI(api_key=token, base_url=base_url)
        self.model_name = model_name

    def call_ai_polish(self, text, cn_prompt=False, prompt=None):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user",
                 "content": ((get_cn_polish_prompt(text) if cn_prompt else get_en_polish_prompt(
                     text)) if prompt is None else prompt.replace('{content}', text))}
            ],
            temperature=0.6,
            # max_tokens=1024,
            stream=False
        )
        # 逐步接收并处理响应
        # for chunk in response:
        #     print(chunk.choices[0].delta.content, end='')
        # print(response.choices[0].message.content)
        content = response.choices[0].message.content
        content = content.split('</think>')[1].strip().replace('\n\n', ' ')
        if text.startswith(r'\par ') and not content.startswith(r'\par '):
            print(rf'missing \par in polished text, append it to the beginning of the text.')
            content = r'\par ' + content
        return content


def polish(main_dir_path, tex_file, server_url, token='Your token here', model_name='deepseek-r1:70b', cn_prompt=False,
           prompt=None, rewrite_list=(r'\caption{', r'\par '), skip_part_list=('figure', 'table', 'equation'),
           ignore_length=100):
    st = Stopwatch()
    ai = AI(token=token, base_url=server_url, model_name=model_name)
    result_tex = merge_content(main_dir_path, tex_file)
    new_tex = ''
    up_flag = False
    result_split = result_tex.split('\n')
    for idx, line in enumerate(result_split):
        spend_count = Stopwatch()
        line = line.strip()
        line_up_flag = True
        if line.startswith('%') or line.startswith('\\') or len(line) < ignore_length:
            for flag in rewrite_list:
                if line.startswith(flag):
                    line_up_flag = False
                    break
        else:
            line_up_flag = False

        for flag in skip_part_list:
            if line.count(r'\begin{' + flag) > 0:
                up_flag = True
                break
            if line.count(r'\end{' + flag) > 0:
                up_flag = False
                break

        if up_flag or line_up_flag:
            new_tex += line + '\n'
        else:
            try:
                print(rf'当前处理{idx}/{len(result_split)}行,总共用时{st.pinch()},当前时间:{get_time_str()}')
                print(f'**p**{line[:100]}{"..." if len(line) > 100 else line}***')
                polish_text = ai.call_ai_polish(line, cn_prompt, prompt)
                print(f'**r**{polish_text[:100]}{"..." if len(polish_text) > 100 else polish_text}***')
                print(rf'处理结束，耗时{spend_count.pinch()},共改写{len(line)}个字符')
                new_tex += polish_text + '\n'
            except Exception as e:
                tqdm.write(f'**e**{e}')
                new_tex += line + '\n'

    with open(r'D:\download\FY_forecast\old.tex', 'w', encoding='utf-8') as f:
        f.write(result_tex)
    with open(r'D:\download\FY_forecast\new.tex', 'w', encoding='utf-8') as f:
        f.write(new_tex)
    write_diff(main_dir_path)


def write_diff(dir_path):
    diff_tex = '''\RequirePackage{shellesc}
    \ShellEscape{pdfLatex new.tex} %编译新文档
    \ShellEscape{pdfLatex old.tex} %编译新文档
    \ShellEscape{latexdiff old.tex new.tex > diff_result.tex}
    \input{diff_result}
    \documentclass{dummy}'''
    with open(rf'{dir_path}\diff.tex', 'w', encoding='utf-8') as f:
        f.write(diff_tex)


if __name__ == "__main__":
    pass
