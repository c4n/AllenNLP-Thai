import re
import glob
import numpy as np
from typing import Dict, List, Iterator
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer 
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer

from allennlp.data.fields import TextField, SequenceLabelField,ArrayField

from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token

import itertools

def is_divider(line):
    return line.strip() == ''

@DatasetReader.register("best2010_reader")
class Best2010Reader(DatasetReader):
    """
    BEST2010 NECTEC (proprietary)
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
                lazy: bool = False) -> None:
        super().__init__(lazy)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(),"token_characters":TokenCharactersIndexer()}
        self.character_token_indexers =  {"ws_tokens": SingleIdTokenIndexer(namespace="ws_tokens")} #for word seg
        self.char_tokenizer=CharacterTokenizer()
    @overrides    
    def text_to_instance(self,char_tokens: List[Token],tokens: List[Token], word_len: List[Token],
                        segment_tags: List[str] = None, 
                         char_pos_tags: List[str] = None,
                         char_ne_tags: List[str] = None,
                         pos_tags: List[str] = None,
                         ne_tags: List[str] = None) -> Instance:
        
        sentence_field = TextField(tokens, self.token_indexers)
        character_field = TextField(char_tokens, self.character_token_indexers)
        wordlen_field = ArrayField(np.array(word_len,dtype=int))
        fields = {"wordseg_file":character_field,"text_file": sentence_field,"word_len":wordlen_field}
        if segment_tags:
            ws_field = SequenceLabelField(labels=segment_tags, sequence_field=character_field, label_namespace = "ws_tags")
            fields["ws_tags"] = ws_field
        if pos_tags and ne_tags:
            pos_field = SequenceLabelField(labels=pos_tags, sequence_field=sentence_field, label_namespace = "pos_tags")
            ner_field = SequenceLabelField(labels=ne_tags, sequence_field=sentence_field, label_namespace = "ne_tags")
            fields["pos_tags"] = pos_field
            fields["ne_tags"] = ner_field
        if char_pos_tags and char_ne_tags:
            pos_char_field = SequenceLabelField(labels=char_pos_tags, sequence_field=character_field, label_namespace="pos_char_tags")
            ne_char_field = SequenceLabelField(labels=char_ne_tags, sequence_field=character_field, label_namespace="ne_char_tags")            
            fields["char_pos_tags"] = pos_char_field
            fields["char_ne_tags"] = ne_char_field            


        return Instance(fields)
        return Instance(fields)
    
    def __cleaner(self,word_rawtext: str) -> str:
        """clean known mistakes after split text into words"""
        word_rawtext=re.sub('/NN//', '/NN/', word_rawtext)
        word_rawtext=re.sub('//PU/', '/PU/', word_rawtext)
        word_rawtext=re.sub('มท.1/NR/__', 'มท.1/NR/ABB_DES_B', word_rawtext)
        word_rawtext=re.sub('MEA_BI', 'MEA_B', word_rawtext)
        return word_rawtext

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        filepath_list = glob.glob(file_path+"/*.txt")
        print(filepath_list)
        for file_path in filepath_list:
            with open(file_path, 'r',encoding="utf-8",errors='ignore') as f:
            #unicodeDecodeError: 'utf-8' codec can't decode byte 0xc0 in position 22: invalid start byte
                text=f.read()
                text_x = [] # textual data
                text_char = []
                text_pos = []
                text_ner = []
                text_char_label = [] #wordseg
                text_char_nelabel = []
                text_char_poslabel = []
                word_len = []
                for word in text.split("|"):
                    if len(word)>1:#if the word is tagged
                        word = self.__cleaner(word)
                        word = re.findall(r'(.*)/(.*)/(.*)',word)     
                        if len(word)<1:#if this pattern is not found it's just a bunch of \n

                            word = ['\n','space','space']# replace multiple \n with just one \n
                        else:
                            word = word[0]
                    else:
                        #they are just empty space
                        word = [word,'space','space']                       

                    if len(word[0])>=1:
                        word = list(word)
                        
                        word[1]=word[1].strip()# clean white space around POS tag
                        word[2]=word[2].strip()# clean white space around NER tag
                        word_len.append(len(word[0]))   
                        text_x.append(word[0])
                        text_pos.append(word[1])
                        text_ner.append(word[2])
                        for i,ch in enumerate(self.char_tokenizer.tokenize(word[0])):
                            text_char.append(ch)
                            text_char_poslabel.append(word[1])
                            text_char_nelabel.append(word[2])                           
                            if i ==0:
                                text_char_label.append('B')
                            else:
                                text_char_label.append('I')
                    else:
                        #Add space when the space is missing between two ||
                        word_len.append(1)   
                        text_char.append(Token(' '))
                        text_x.append(' ')
                        text_pos.append(word[1])
                        text_ner.append(word[2]) 
                        text_char_label.append('B')
                        text_char_poslabel.append(word[1])
                        text_char_nelabel.append(word[2])
                 
                yield self.text_to_instance(text_char,[Token(word) for word in text_x],word_len,
                                            text_char_label,text_char_poslabel,text_char_nelabel,
                                            text_pos, text_ner)
