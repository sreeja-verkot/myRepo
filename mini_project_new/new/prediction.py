from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

#preprocessing
def read_file(file_name): 
    data = []
    with open(file_name, 'r') as f: 
        for line in f: 
           #print(line)
            line = line.strip() 
            #print(line)
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            #print(label,text)
            data.append([label, text])
    return data 
def create_feature(text, nrange=(1, 1)):
    text_features = [] 
    text = text.lower() 

    # 1. treat alphanumeric characters as word tokens
    # Since tweets contain #, we keep it as a feature
    # Then, extract all ngram lengths
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1): 
        text_features += ngram(text_alphanum.split(), n)
    
      
    # 2. Return a dictinaory whose keys are the list of elements 
    # and their values are the number of times appearede in the list.
    return Counter(text_features)
def ngram(token, n): 
    output = []
    for i in range(n-1, len(token)): 
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram) 
    return output

file_name = "psychExp.txt"
psychExp_txt = read_file(file_name)
#print(psychExp_txt[0:3])

#label generation
emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
for i in range(len(psychExp_txt)):
    item_new=psychExp_txt[i][0].replace('.','')
    item_new=item_new.replace(" ",'')
    #print(item_new)
    index=item_new.find("1")
    #print(index)
    label=emotions[index]
    #print(label)
    psychExp_txt[i][0]=label
#print(psychExp_txt)

X=[]
y=[]
for label, text in psychExp_txt:
    y.append(label)
    X.append(text)
#print(X,y)
emotion_encoder = LabelEncoder()
y = emotion_encoder.fit_transform(y)
print(list(emotion_encoder.classes_))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
#print(X_train,y_train)
vect = CountVectorizer()
vect.fit(X_train)
#print(vect.get_feature_names())
X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)

svc=SVC() 
svc.fit(X_train_dtm,y_train)
y_pred=svc.predict(X_test_dtm)
print('\nAccuracy Score:\n')
print(metrics.accuracy_score(y_test,y_pred))
input_msg=input("enter a message\n")
print(svc.predict(vect.transform([input_msg])))
