#!/usr/bin/python

"""
Construct probabilistically populated patient-specific templates

(C) 2013 GNU General Public License v3.

Authors: Rohit Joshi, Tristan Naumann, Marzyeh Ghassemi, Andreea Bodnari

"""
import re
import math
import os
import collections
from operator import itemgetter

# external libraries
# install nltk
import nltk.data


section_tags = [‘Assessment’, ‘Diagnosis’, ‘Interventions’, ‘Rationale’, ‘Evaluation’]
careplan_tags = [ ‘Diabetes Mellitus Type II’, ‘Glaucoma’, ‘Major Depressive Disorder’, ‘AIDS’, ‘renal kidney disease’] # major patient conditions
orderset_tags = [Antibiotics, common_IV_drips, labs_diagnostic_tests]
#contexts = [section_tags, careplan_tags, orderset_tags, ...]

templates = {}
# add pre-existing templates
for careplan in careplan_tags:
	for section in section_tags:
        templates.setdefault((careplan,section),[]).append( [(list_template_actions_sentences, likelihood)])


#########################################################
# load the dictionary of documents from the os path
docid = 0
documents={} # dictionary of docids and words in each document
doclist={} # dictionary of docids and filepath
patient_records={} # dictionary of patient id and document ids
rootpath=”/scratch/mimic/data/amia/”

# put all patient notes in one folder OR in one file 
for folder in os.listdir(rootpath):
    patientid += 1
    if os.path.isdir(folder) == True:
        folderpath= os.path.join(rootpath, folder)

        for ptfile in os.listdir(folderpath): 
            docid+=1
            documents.setdefault(docid,[]).append(getwords(ptfile))
            doclist.setdefault(docid,[]).append(os.path.join(folderpath,ptfile))
            patient_records.setdefault(patientid,[]).append(docid)
	else:
        docid+=1
        documents.setdefault(docid,[]).append(getwords(ptfile))
        doclist.setdefault(docid,[]).append(os.path.join(rootpath,ptfile))
        patient_records.setdefault(patientid,[]).append(docid)


################################################################


#######################################
## EXTRACT DOCUMENT INFORMATION : Sections, Sentence templates, Scores ###################################################
document_contexts={}
disease_orders={}
ngram_dictionary={}
 # build the freq counts of (careplan, section, ngrams)

# extract the best sentence

for docid, doc in documents.items():
    careplans = identify_careplan_tags(doc)
    sentences = identify_sentences(docid)
    sentence_section = {}

    for sentence in sentences:
        sec = identify_sections(doc,sentence) # assume one section per sentence 
        sentence_section.setdefault(sentence,[]).append(sec)

        # compute the copy-paste score of possible templates of each sentence
        # given patient careplan tag and section
        # use greedy approach: use the sentence_template 
        # (sentence - deleted(words_in_sentence)) and keep the highest scoring 
        # template, return the template and add the template to the score dictionary

        (careplantag, sec, sentemplate, senscore) = best_copy_paste_score(sentence, careplans, sec)

        # append new learned sentence templates

        templates.setdefault((careplantag, sec), {})
		templates[(careplantag, section)].setdefault(sentemplate,0.0)
		templates[(careplantag, section)][sentemplate] = senscore
	
    # get unique values
    sections = set(sentence_section.values()) 

    ordersets = identify_orderset_tags(doc)

    # link careplans to order sets
    for careplan in careplans:
	    for order in ordersets:
            careplan_orders.setdefault((careplan,order),0)
            careplan_orders[(careplan,order)] +=1
 
    document_contexts.append(docid, [sentence-sections, sections, careplans, ordersets])

def ngrams_nearby_sentences(sentence, n=3, deletewords=1):
	words = sentence.split()
	wordsclone = words.copy()
	sentence_variations=[]
	
	for indx, word in enumerate(words):
        del(wordsclone[word])
        wordsclone2=wordsclone.copy()
		if deletewords == 2: # improve logic to delete > 2 words
			for nextword in words[indx+1:]:
				del(wordsclone2[nextword])
				newsentence=’ ‘.join(wordsclone2)
				sentence_variations.append(create_ngrams(newsentence, n))
				wordsclone2=wordsclone[:]
		else:
            newsentence=’ ‘.join(wordsclone)
			sentence_variations.append(create_ngrams(newsentence, n))
				
		wordsclone=words[:]

    return sentence_variations

def create_ngrams(sentence, n=3):
	ngrams=[]
    for i in enumerate(sentence.split()):
		ngram=zip([sentence[i+j] for j in range(0,n-1)])
		ngrams.append(ngram)
	return ngrams

def copy_paste_score(sentence_ngrams, careplan, section):
	score=0.0
	for ngram in sentence_ngrams:
	    score+= math.log(ngram_dictionary[(careplan, section, ngram)] / totalwords(ngram_dictionary))
	return score

def totalwords(dictionary):
	return sum(dictionary.values())

def best_copy_best_score(sentence, careplans, section, MAXDELETE = 2):
	bestscore=0.0; careplantag=’’
    for careplan in careplans:
        currentscore = copy_paste_score(create_ngrams(sentence), careplan, section)
        if currentscore > bestscore: 
            bestscore = currentscore
            careplantag = careplan
	
    newsentence = sentence
	besttemplate = sentence
	
	for i in range(1,MAXDELETE):
        sentence_variations = ngrams_nearby_sentences(newsentence, deletewords=i)
        for sen_ngram in sentence_variations:
	        sentemplate=’ ’.join(sen_ngram)
            currentscore= copy_paste_score(sen_ngram, careplans, sec)
            if currentscore > bestscore:
	            bestscore = currentscore
	            besttemplate=sentemplate

	return (careplantag, section, besttemplate, bestscore)


# tokenize files
def getwords(file):
    for line in open(file):
		words = [word.strip() for word in line.split(‘ ‘) if word not in stopwords]
    return words

def identify_sentences(docid):
	docpath=doclist[docid]
	# can also use ctakes / opennlp for better sentence breaking
	# using nltk python nlp toolkit
    sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    fp = open(docpath)
    text = fp.read()
    sentences = sent_tokenizer.tokenize(text)
	return sentences

def identify_disease_tags(doc):
	tags = [word for word in doc if word in disease_tags]
 	return tags

def identify_orderset_tags(doc, diseases):
	tags = [word for word in doc if word in orderset_tags ]
 	return tags

def editdistance(sentence1, sentence2):
	return None
		

########### QUERY ###############
stopwords = getwords(“\path\to\stopwordfile”)
queryvector = getwords(“\path\to\querydocpath”)
wordlist = {}
### documents / careplans similarity
 for word in queryvector:
# create a list of possible care plans using n-grams
wordlist.setdefault(word, int(len(dict)+1))
wordid = wordlist[word]
	for careplan in careplan_tags:
        score.setdefault(careplan,0.0)
		score[careplan] += math.log(prob(wordid,careplan))

# find the best matching careplan scores 
sorted_careplans = sorted(careplan, key=careplan.get, reverse=True)
# sorted_careplans = sorted(careplan, key=operator.itemgetter(1), reverse=True)

# find similar documents using cosine similarity score
for docid in documents:
    cos_score.setdefault(docid, 0.0)
    # need to parallelize this step
    cos_score[docid] = cosine_similarity(queryvector, documents[docid])  

# list of best matching documents
sorted_cos_scores = sorted(cos_score, key=operator.itemgetter(1), reverse=True)

# extract (section, [(sentence templates, scores)]) from top n best matching documents and # sort
sentence_templates = template[(careplan,section)]
sorted_sentence_templates = sorted(sentence_templates, operator.itemgetter(1), reverse=True)

# for each section; write out the top k sentence templates to a file
topk=5
with open(writepath) as f:
	for sen in sorted_sentence_templates[:topk]
		f.write(sen + ‘\n’)


# cosine similarity function
def cosine_similarity(vec1, vec2):
	score = 0.0
	len_v1 = len(vec1)
	len_v2 = len(vec2)
	for v1 in vec1:
		if v1 in vec2: score+=1.0
 	score = score /(len_v1 * len_v2)
return score
