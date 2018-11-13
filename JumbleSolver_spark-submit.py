#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 15:18:02 2018

@author: anassar
""" 

import pandas as pd
import time
import uuid
import json

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

conf = SparkConf().setAppName('Jumble Solver')
pc = SparkContext(conf = conf)
qc = SQLContext(pc)
df = qc.read.csv(path = 'data/freq_dict.csv', sep = ':', header = True,
                 ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)


high_freq_dict = 'high_freq_dict'
low_freq_dict = 'low_freq_dict'

limit_dictionary_expantion = 7


print('starting jumble solver')
pdf = df.toPandas()
pdf['count'] = pdf['count'].apply(lambda x : int(x[:-1]))
pdf['hash'] = pdf['word'].apply(lambda x: hash(''.join(sorted(x))))

#dictionary building are offline  and one time
#let group on words length, we are going to need it
pdf['word_length'] = pdf['word'].apply(lambda x : len(x))
pdf = pdf.sort_values(by='hash')

high_freq_pdf = pdf[pdf['count'] > 0]
high_freq_pdf = high_freq_pdf.groupby(by = 'word_length')

low_freq_pdf = pdf[pdf['count'] == 0]
low_freq_pdf['count'] = 500000
low_freq_pdf = low_freq_pdf.groupby(by = 'word_length')

pdf_dict = {}
pdf_dict[high_freq_dict] = high_freq_pdf
pdf_dict[low_freq_dict] = low_freq_pdf

pdf = pdf.groupby(by = 'word_length')

print('finish dictionary preparation')

def find_partialy_match(row, word):
    found = True
    remain = word
    for c in row['word']:
        indx = remain.find(c)
        if(indx < 0):
            found = False
            break
        remain = remain[:indx] + remain[indx+1:]
    row['found'] = found
    row['remaining'] = remain
    return row

def lookup_full_match(pdf_dict, word):
    word_hash = hash(''.join(sorted(word)))
    #look up the high freq first
    if(len(word) in pdf_dict.get(high_freq_dict).groups.keys()):
        mask = pdf_dict.get(high_freq_dict).get_group(len(word))['hash'].values == word_hash
        assemble_list = pdf_dict.get(high_freq_dict).get_group(len(word))[mask]
        if(len(assemble_list) > 0):
            return assemble_list
    
    # remove the lookup of the low freq dictionary for performance 
    if(len(word) in pdf_dict.get(low_freq_dict).groups.keys()):
        mask = pdf_dict.get(low_freq_dict).get_group(len(word))['hash'].values == word_hash
        assemble_list = pdf_dict.get(low_freq_dict).get_group(len(word))[mask]   
        if(len(assemble_list) > 0):
            return assemble_list
    
    return pd.DataFrame()

def lookup_partialy_match(pdf_dict, word, word_length):
    if(word_length == len(word)):
        results = lookup_full_match(pdf_dict, word)
        if(len(results) > 0):
            results['remaining'] = ''
        return results
    #word = word_copy.copy()
    #look up the high freq first
    if(word_length in pdf_dict.get(high_freq_dict).groups.keys()):
        assemble_list = pdf_dict.get(high_freq_dict).get_group(word_length).apply(find_partialy_match,
                                 args = (word,), axis=1)
        #assemble_list = assemble_list[assemble_list['found'] == True]
        #mask = assemble_list['found'] == True
        assemble_list = assemble_list.loc[assemble_list['found'] == True]
        if(len(assemble_list) > 0):
            assemble_list = assemble_list.sort_values(by = 'count', ascending=True)
            return assemble_list
    
    if(word_length in pdf_dict.get(low_freq_dict).groups.keys()):
        assemble_list = pdf_dict.get(low_freq_dict).get_group(word_length).apply(find_partialy_match,
                                 args = (word,), axis=1)
        #assemble_list = assemble_list[assemble_list['found'] == True]
        assemble_list = assemble_list.loc[assemble_list['found'] == True]
        if(len(assemble_list) > 0):
            assemble_list = assemble_list.sort_values(by = 'count', ascending=True)
            return assemble_list
    
    return pd.DataFrame()

#c_test = sortCandidates([('11',11),('5',5),('7',7)])
def sortCandidates(candidates):
    cand_dict = {}
    for cand in candidates:
        if(cand[1] not in cand_dict.keys()):
            cand_dict[cand[1]] = []
        cand_dict.get(cand[1]).append(cand)
    cand_score = sorted(cand_dict.keys())
    sorted_list = []
    for s in cand_score:
        for c in cand_dict.get(s):
            sorted_list.append(c)
    return sorted_list

def expand_match_dict(wk,level, vl):
    if(len(wk.keys()) == level):
        return vl
    temp_list = []
    token = wk.get(level)
    if level == 0:
            temp_list = [([token[0]], token[1])]
    else:
       for vt in vl:
           tl = vt[0].copy()
           tl.append(token[0])
           tv = token[1]+vt[1]
           temp_list.append((tl, tv))
    vl = temp_list
    return expand_match_dict(wk, level+1, vl)

#test_results = expand_full_match(mr_test)
def expand_full_match(full_matched):
    verified_list = []
    for k, wk in full_matched.items():
        temp_list = []
        verified_list.append(expand_match_dict(wk,0, temp_list))
        verified_list.extend(temp_list)
    result = []
    for item in verified_list:
        for sItem in item:
            result.append(sItem)
    return result

def find_best_match(verified_list):
    v_dict = {}
    for v in verified_list:
        if v[1] not in v_dict.keys():
            v_dict[v[1]] = []
        v_dict[v[1]].append(v)
    sorted_key = sorted(v_dict.keys())
    results = []
    for s in sorted_key:
        for item in v_dict.get(s):
            results.append(item)
    return results

def get_final_candidates(pdf_dict, remaining, score, level, target_list, matched_list, prune_list, full_list):
    if(level == len(target_list)):
        sId = str(uuid.uuid1())
        full_list[sId] = matched_list
        return True
    if(remaining in prune_list):
        return False
    prune_list.append(remaining)
    m_list = lookup_partialy_match(pdf_dict, remaining, target_list[level])
    found = False
    #this is point of congestion, splict here will be increase the performance
    count = 0 # limit it to only 5 remainings
    for indx, row in m_list.iterrows():
        temp_match = matched_list.copy()
        temp_match[level] = ((row['word'], score + row['count']))
        r = get_final_candidates(pdf_dict, row['remaining'], score+row['count'], level+1, target_list, temp_match, prune_list, full_list)
        prune_list.append(remaining)
        if(r == False):
            continue
        
        count = count + 1
        found = True
        if(level == 0):
            print('progress -- ', (count/(limit_dictionary_expantion+2)*100))
        if(count >= limit_dictionary_expantion):
            break
    return found
    
def solve(problem, pdf_dict):
    candidates = []
    candidates.append(('', 0))
    #this is point of congestion, splict here will be increase the performance
    
    for jumble in problem.get('jumble'):
        word = jumble.get('word')
        print('full match word -', word)
        assemble_list = lookup_full_match(pdf_dict, word)
        for index, row in assemble_list.iterrows():
            print('matched -', row['word'])
        print('-----------')
        for loc in jumble.get('locations'):
            ch_list = []
            temp_candidates = []
            for index, row in assemble_list.iterrows():
                ch = row['word'][loc]
                score = row['count']
                if ch in ch_list:
                    continue
                ch_list.append(ch)
                for cand in candidates:
                    temp_candidates.append((cand[0]+ch, cand[1]+score))
            candidates = temp_candidates
            
    #candidates.append(('trivtsneetsdel', 4513182))
    print('candidates')
    for cand in candidates:
        print(cand)
    print('##################')
    #let's loop over every candidate and verify if it is the right one
    verified_list = []
    candidates = sortCandidates(candidates)
    for cand in candidates:
        #cand length should be equals to target sum
        target_list = problem.get('target')
        if(len(cand[0]) != sum(target_list)):
            print('serious problem candidate lenth', len(cand[0]), 'does not match the target sum', sum(target_list))
            continue
        if(len(target_list) <1): #there is no target to acheive
            continue
        full_matched = {}
        matched_list = {}
        prune_list = []
        prune_list.append('')
        get_final_candidates(pdf_dict,cand[0], cand[1], 0, target_list, 
                             matched_list, prune_list, full_matched)
        print('finish final candidates -- ',len(full_matched))
        #expand all full matched if we got one
        if(len(full_matched) > 0):
            verified_list.extend(expand_full_match(full_matched))
    return find_best_match(verified_list)
    #return None


def print_save_results(r):
    with open('./data/results.json', 'w') as fp:
        json.dump(r, fp, indent=4)
    for key, item in r.items():
        print('results for puzzel', key)
        print('---------------------')
        for sItem in item:
            print(sItem)
    
def main(results):
    problems = pd.read_json('./data/problems.json', orient='records')
    counter = 0
    for problem in problems.get('problem_list'):
        start_time = time.time()
        counter = counter + 1
        r = solve(problem, pdf_dict)
        if(r is not None):
            solution = {}
            solution['solutions'] = r
            solution['elapsed_time'] = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            results[problem.get('name')] = solution
    print_save_results(results)

results = {}
main(results)