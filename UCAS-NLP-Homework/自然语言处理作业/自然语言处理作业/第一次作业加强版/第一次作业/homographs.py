from docx import Document
import jieba
from pypinyin import pinyin
import re

def is_all_punctuation(s):#用于正则匹配中英文字符
    # 正则表达式匹配中英文标点符号
    # 包括了常见的中文标点和英文标点
    punctuation_regex = r'^[\u3000-\u303F\uFF00-\uFFEF\uff5c-\uff5e\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65，。？！、；：‘’“”（）《》【】—…-]+$'
    return bool(re.fullmatch(punctuation_regex, s))


dict_word_pinyin = {}  #用于保存字与拼音的dict

# 获取多音字与对应拼音，放入dict_word_pinyin
def get_heteronym(text):
    # dict_word_pinyin = {}  #用于保存多音字与拼音的dict
    list_jieba_split_word = jieba.lcut(text)#将这段的文字分割为各个词语，用dict保存

    # 处理多音字
    for words in list_jieba_split_word: #找到这段文字所有的多音字，并保存索引和拼音
        # print(words,len(words))
        if words.isnumeric() == True :#这个词语是数字，不用标注拼音
            # print(words,'是一个数字')
            pass
        elif is_all_punctuation(words) == True:#这个词语是中英文字符，不用标注拼音
            # print(words,'是一个中英文字符')
            pass
        else:#这个是词语，需要标注拼音
            for index in range(len(words)):
                if len(pinyin(words,heteronym=True)[index]) > 1 :
                    # print(words[index],"是一个多音字",words[index],'的拼音是:',pinyin(words,heteronym=True)[index][0])
                    dict_word_pinyin.update({words[index]:pinyin(words,heteronym=True)[index][0]})
                else:
                    # print(words[index],"不是一个多音字")
                    pass
                # print(words,pinyin(words,heteronym=True),end=' ')
    #end for words in lst_split_word

    
    # for key,value in dict_word_pinyin.items():
    #     modified_text = modified_text.replace(key, f'{key}【{value}】')

    # return modified_text
#end def text_to_heteronym(text):

# 获取生僻字与对应拼音，放入dict_word_pinyin
def get_spz(text):
    modified_text = ''
    spz_list = []
    with open('生僻字3.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            line.strip() # 去掉末尾换行符
            spz_list.append(line.strip())
            i= i+1
        # print(spz_list)
        # print(i/2)# 6318

    # 使用字典推导式实现转换
    spz_dict = {spz_list[i+1]: spz_list[i] for i in range(0, len(spz_list) - 1, 2)}

    #遍历文章的每一个字，并查询生僻字表，如果这个字是生僻字，加入生僻字dict
    for index in range(len(text)):
        # print(modified_text[index],end=' ')
        if text[index] in spz_dict :
            # print(f'{index}{text[index]}【{spz_dict[text[index]]}】')
            dict_word_pinyin.update({text[index]:spz_dict[text[index]]})
            # modified_text = text.replace(text[index],f'{text[index]}【{spz_dict[text[index]]}】')

    # for key,value in dict_spz_pinyin.items():
    #     modified_text = text.replace(key, f'{key}【{value}】')

    # return modified_text


# 返回更改后的文本
def change_text(text):
    modified_text = text
    get_spz(text)
    get_heteronym(text)

    for key,value in dict_word_pinyin.items():
        modified_text = modified_text.replace(key, f'{key}【{value}】')

    return modified_text


#打开一个word
doc = Document('test3.docx')

for para in doc.paragraphs:
    para.text = change_text(para.text)
    dict_word_pinyin.clear()


#标注多音字后的text 保存到新的docx
doc.save('test31.docx')
