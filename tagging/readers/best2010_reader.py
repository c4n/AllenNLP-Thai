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
    @overrides    
    def text_to_instance(self,tokens: List[Token], 
                         pos_tags: List[str] = None,
                         ne_tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}
        if pos_tags and ne_tags:
            #pos_field = SequenceLabelField(labels=pos_tags, sequence_field=sentence_field, label_namespace = "pos_tags")
            ner_field = SequenceLabelField(labels=ne_tags, sequence_field=sentence_field, label_namespace = "ne_tags")
	    #fields["pos_tags"] = pos_field
            fields["ne_tags"] = ner_field


        return Instance(fields)
        return Instance(fields)
    
    def __cleaner(self,word_rawtext: str) -> str:
        """clean known mistakes after split text into words"""
        word_rawtext=re.sub('/NN//', '/NN/', word_rawtext)
        word_rawtext=re.sub('//PU/', '/PU/', word_rawtext)
        word_rawtext=re.sub(r'มท.*/NR/__', 'มท.1/NR/ABB_DES_B', word_rawtext)
        word_rawtext=re.sub('MEA_BI', 'MEA_B', word_rawtext)
        word_rawtext=re.sub('/EA_I', '/MEA_I', word_rawtext)
        word_rawtext=re.sub('อาร์พี/NR/ABB', 'อาร์พี/NR/O', word_rawtext)
        word_rawtext=re.sub('หนึ่ง/NN/DDEM', 'หนึ่ง/DDEM/O', word_rawtext)

        return word_rawtext

    def __span_label_pattern(self,word_rawtext: str) -> str:
        """change label pattern so that it matches span-f1"""
        word_rawtext=re.sub(r'(.*)_B',r'B-\1',word_rawtext)
        word_rawtext=re.sub(r'(.*)_I',r'I-\1',word_rawtext)
       
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
                text_pos = []
                text_ner = []
                word_len = []
                for word in text.split("|"):
                    if len(word)>1:#if the word is tagged
                        word = self.__cleaner(word)
                        word = re.findall(r'(.*)/(.*)/(.*)',word)     
                        if len(word)<1:#if this pattern is not found it's just a bunch of \n

                            word = ['\n','space','O']# replace multiple \n with just one \n
                        else:
                            word = word[0]
                    else:
                        #they are just empty space
                        word = [word,'space','O']                       

                    if len(word[0])>=1:
                        word = list(word)
                        
                        word[1]=word[1].strip()# clean white space around POS tag
                        word[2]=word[2].strip()# clean white space around NER tag
                        word[2]=self.__span_label_pattern(word[2])
                        word_len.append(len(word[0]))   
                        text_x.append(word[0])
                        text_pos.append(word[1])
                        text_ner.append(word[2])
                    else:
                        #Add space when the space is missing between two ||
                        word_len.append(1)   
                        text_x.append(' ')
                        text_pos.append(word[1])
                        text_ner.append(self.__span_label_pattern(word[2])) 
                 
                yield self.text_to_instance([Token(word) for word in text_x],
                                            text_pos, text_ner)
