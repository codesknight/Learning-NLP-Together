from docx import Document
from pypinyin import pinyin
import jieba
import re


def is_all_punctuation(s):
    # 正则表达式匹配中英文标点符号
    # 包括了常见的中文标点和英文标点
    punctuation_regex = r'^[\u3000-\u303F\uFF00-\uFFEF\uff5c-\uff5e\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65，。？！、；：‘’“”（）《》【】—…-]+$'
    return bool(re.fullmatch(punctuation_regex, s))

def text_to_heteronym(text):
    modified_text = text#用于保存新的多音字
    dict_word_pinyin = {}  #用于保存多音字与拼音

    lst_split_word = jieba.lcut(text)#将这段的文字分割为各个词语
    for words in lst_split_word: #找到这段文字所有的多音字，并保存索引和拼音
        # print(words,len(words))
        # print('#'*50)
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
    
    for key,value in dict_word_pinyin.items():
        modified_text = modified_text.replace(key, f'{key}【{value}】')
    # print(modified_text)
    return modified_text
#end def text_to_heteronym(text):

#打开一个word
doc = Document('test3.docx')

for para in doc.paragraphs:
    para.text = text_to_heteronym(para.text)

#标注多音字后的text 保存到新的docx
doc.save('test3_2.docx')