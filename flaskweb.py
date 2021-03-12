from flask import request, abort, current_app as app
from flask import Flask
import re
import time
import pickle
import numpy as np
import requests
from sklearn.preprocessing import Normalizer

response = 0
PLAYER_NAME = []
returned_data = []
newplayername = []
newplayerskills = []
DATAarray = []
userfixed = ""
osrsknn_predict = -1

tempnames = list()
tempgroups = []
tempind = 0

MAIN = "http://services.runescape.com/m=hiscore_oldschool/index_lite.ws?player="

osrsknn = pickle.load(open("OSRS_KNN_V1","rb"))
y_km = pickle.load(open("ykmfile","rb"))
PLAYER_TRAIN = pickle.load(open("traindata","rb"))
player_name = pickle.load(open("pnamefile","rb"))
PLAYER_IND = pickle.load(open("PIfile","rb"))

normal = Normalizer()
PLAYER_IND = np.reshape(PLAYER_IND,(-1,78))
PLAYER_IND = normal.fit_transform(PLAYER_IND)

for name in player_name:
    PLAYER_NAME.append(name.replace('\n',''))
    
##################################################################################################

app = Flask(__name__)

@app.route('/', methods =['POST'])
def post():
    if request.method == 'POST':
        print("Someone attempted to post")
    return 
        
@app.route('/user/<user>', methods =['GET'])
def get(user):
    global osrsknn_predict
    userfixed = user.replace(' ','_')
    print(userfixed)
    if userfixed in PLAYER_NAME:
        ind = PLAYER_NAME.index(userfixed)
        osrsknn_predict = osrsknn.predict(PLAYER_TRAIN[ind].reshape(1,-1))
        player_predprob = osrsknn.predict_proba(PLAYER_TRAIN[ind].reshape(1,-1))
        print(osrsknn_predict)
        print(player_predprob)
        print(y_km[ind])
        print(PLAYER_NAME[ind])
        return str(getResponse(osrsknn_predict)).strip('[]')
    else:
        if userfixed in tempnames:
            tempind = tempnames.index(userfixed)
            print("Name and Group taken from Storage.")
            return str(tempgroups[tempind]).strip('[]')
        else:
            try:
                print("User not found. Currently evaluating user...")
                pulldata(userfixed)
                return str(getResponse(osrsknn_predict)).strip('[]')
            except: 
                print("CODE ERROR")
    return 
        
def pulldata(userfixed):
    global osrsknn_predict
    global tempnames
    url = MAIN+userfixed 
    response = requests.get(url) 
    data = response.text 
    try: 
        if data.find('404 - Page not found') != -1:
            osrsknn_predict = -1
        else:
            r = str.split(data) 
            DATAarray = [[float(n) for n in row.split(",")] for row in r]
            cleanup(DATAarray)
            tempnames.append(userfixed)
            print("Name list",tempnames)
    except: 
        print("INTERNAL ERROR")
    return

def cleanup(DATAarray):
    global normal
    global newplayerskills
    DATAcheck = np.asarray(DATAarray)
    
    for x in range(0,len(DATAcheck)):
        if (0<x<24):
            newplayerskills = np.append(newplayerskills,DATAcheck[x][2])
        if (24<x<80):
            newplayerskills = np.append(newplayerskills,DATAcheck[x][1])

    newplayerskills = np.asarray(newplayerskills.reshape(-1, 78))
    newplayerskills = normal.transform(newplayerskills)
    
    print(newplayerskills)
    print("DATA CLEANED FOR USER")
    osrsKNN(newplayerskills)
    
    newplayerskills = []
    return 

def osrsKNN(newplayerskills):
    global osrsknn_predict
    print("SCANNING...")
    osrsknn_predict = osrsknn.predict(newplayerskills.reshape(1, -1))
    player_predprob = osrsknn.predict_proba(newplayerskills)
    print("Group: "+str(osrsknn_predict))
    print("Player Grouping Data: "+str(player_predprob))
    print("PLAYER SUCCESSFULLY SCANNED")
    tempgroups.append(osrsknn_predict)
    print(tempgroups)
    return 

def getResponse(osrsknn_predict):
    return osrsknn_predict

if __name__ == '__main__':
    app.run(debug=True)
