import json
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
with open('/home/user/Desktop/mtech sem1/lab2/affix.json') as f:
    data = json.load(f)
def check(file_name,word):
    datafile=open(file_name).read()
    found = False #this isn't really necessary 
    for line in datafile.split("\t"):
        if word == line:
            #found = True #not necessary 
            return True
    return False

with open('/home/user/Desktop/mtech sem1/lab2/file.txt') as f:
    content = f.readlines()
    words=[]
    for line in content:
        tokenz=word_tokenize(line)
        words.extend(tokenz)
    stopWords= list(stopwords.words('english'))
    filtered_words=[]
    for w in words:
        if w not in stopWords:
            filtered_words.append(w)
    print(filtered_words)
    choice="0"
    while(choice!='3'):
        choice=input("\nEnter the choice: \n1-prefix identification\t 2- suffix identification\t 3- end\n")
        if choice=='1':
            file = open('pre.txt', 'a+')
            prefix=data.get('prefix')
            print(prefix)
            for p in prefix:
                for w in filtered_words:
                    if w.startswith(p):
                        result=check('pre.txt',w)
                        print(w,result)
                        if result==False:
                            print(w)
                            file.write(w +"\t")
        if choice=='2':
            file = open('suf.txt', 'a+')
            suffix=data.get('suffix')
            print(suffix)
            for s in suffix:
                for w in filtered_words:
                    if w.endswith(s):
                        result=check('suf.txt',w)
                        print(w,result)
                        if result==False:
                            print(w)
                            file.write(w +"\t")

    
    
