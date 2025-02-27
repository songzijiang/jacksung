import os
from openai import OpenAI
from tqdm import tqdm


def get_polish_prompt(content):
    polish_prompt = \
        f'''
        用学术写作风格重写下面的文本,在保持原本涵义不变的情况下使用更合适的词汇和句子结构:
        Text:
        ---------
        {content}
        ---------
        确保改写后的版本传达的信息和意图与原文相同。
        请直接以latex格式输出重写后的文本，不需要包含原文和思考逻辑等其他内容。
        注意特殊符号和公式以latex格式输出，而不是直接输出特殊字符。
        Response:
        '''
    return polish_prompt


def merge_content(tex_dir, main_tex):
    result_tex = ''
    with open(os.path.join(tex_dir, main_tex), 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith(r'\input{'):
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

    def call_ai_polish(self, text, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user",
                 "content": (get_polish_prompt(text) if prompt is None else prompt.replace('{content}', text))}
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
        return content


def polish(main_dir_path, tex_file, server_url, token='Your token here', model_name='deepseek-r1:70b', prompt=None):
    ai = AI(token=token, base_url=server_url, model_name=model_name)
    result_tex = merge_content(main_dir_path, tex_file)
    new_tex = ''
    up_flag = False
    for line in tqdm(result_tex.split('\n')):
        line = line.strip()
        flag_list = ['figure', 'table', 'equation']
        for flag in flag_list:
            if line.count(r'\begin{' + flag) > 0:
                up_flag = True
            if line.count(r'\end{' + flag) > 0:
                up_flag = False
        if not line.startswith(r'\caption{') and (
                line.startswith('%') or line.startswith('\\') or line == '' or up_flag):
            new_tex += line + '\n'
        else:
            tqdm.write('polish:' + line[:100])
            polish_text = ai.call_ai_polish(line, prompt)
            tqdm.write('polish result:' + polish_text[:100])
            new_tex += polish_text + '\n'

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
