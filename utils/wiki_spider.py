""" Get Gene Wikipedia text information from Gene Card

"""
# pip install requests

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os

label_space = 'react'
label_space = 'oncokb' # use this

data = 'tcga'

def list_txt(path, list=None):
    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist

headers = {'user-agent': 'Mozilla/5.0'}

list_path = './data/processed data/list'
gene_list = list_txt(path=list_path + '/{}_{}_inter_gene_list.txt'.format(data, label_space))
# gene_list = ['TP53', 'PIK3CA']
print(len(gene_list))
gene_card_pd = pd.DataFrame(columns=['Entrez Gene Summary', 'CIViC Summary', 'GeneCards Summary', 'UniProtKB/Swiss-Prot Summary'])
col = gene_card_pd.columns.values.tolist()
for count, gene in enumerate(gene_list):
    print("Processing Gene [{}] ".format(gene))
    # url_path = 'https://www.genecards.org/cgi-bin/carddisp.pl?gene={}'.format(gene)
    url_path = 'https://www.genecards.org/cgi-bin/carddisp.pl?gene={}#summaries'.format(gene)
    print(url_path)
    header_set = ['Entrez Gene Summary for {} Gene'.format(gene),
                'CIViC Summary for {} Gene'.format(gene),
                'GeneCards Summary for {} Gene'.format(gene),
                'UniProtKB/Swiss-Prot Summary for {} Gene'.format(gene)]

    # initialize data
    new_row = pd.Series({key:0 for key in col}, name = gene)
    gene_card_pd = gene_card_pd.append(new_row)

    req = requests.get(url = url_path, headers=headers)
    # , headers = header_set[i])
    req.encoding = 'utf-8'
    html = req.text
    soup = BeautifulSoup(html, features='html.parser')
    sections = soup.find_all("div",class_="gc-subsection")
    # 
    # headers belong to h3
    # class = "gc-subsection-header"
    # count = 0    
    for i in range(len(header_set)):
        for c, sec in enumerate(sections):
            h = sec.find_all("div",class_='gc-subsection-header')
            if len(h) == 0:
                pass
            else:
                print(sec.div.h3.text.strip())                
                if sec.div.h3.text.strip() == header_set[i]:    
                    if len(sec.find_all("p")) != 0:
                        if len(sec.find_all("ul")) != 0:
                            text = sec.ul.li.p.text.strip()                            
                        else:
                            text = sec.p.text.strip()                            
                        text =' '.join(text.split())
                    else:
                        text = 0
                    
                    # if i>=2:
                    #     text = sec.p.text.strip()
                    # else:
                    #     text = sec.ul.li.p.text.strip()

                    gene_card_pd.iloc[count,i] = text
                    break

print(len(gene_card_pd))
gene_card_pd.to_csv(os.path.join(list_path,'{}_{}_gene_card_embed.csv'.format(data, label_space)))
        # print(req.text)