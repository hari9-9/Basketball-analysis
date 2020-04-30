import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import csv
import itertools

merits={}
train=pd.read_csv('/Users/MAHE/Desktop/programming/machinelearning/basketball/shot_logs.csv')
train.dropna()
train= train[(train['player_name']=='stephen curry') | (train['player_name']=='klay thompson')]

# print(train.head())
print(train.dtypes)
# plt.scatter(train['SHOT_DIST'],train['CLOSE_DEF_DIST'],c=train['FGM'])
# plt.title('field goal WRT shot distance and closest defender(combined)')
# plt.xlabel('Shot distance')
# plt.ylabel('Closest Defender distance')
# plt.show()

# plt.scatter(train['SHOT_DIST'],train['TOUCH_TIME'],c=train['FGM'])
# plt.title('field goal WRT shot distance and Touch time(combined)')
# plt.xlabel('Shot distance')
# plt.ylabel('Touch time')
# plt.show()

#splittig data player wise
curry_data = train[(train['player_name']=='stephen curry')]
klay_data = train[(train['player_name']=='klay thompson')]

#overall shooter
count_makes_curry=len(curry_data[curry_data['FGM']==1])
count_misses_curry=len(curry_data[curry_data['FGM']==0])
fgm_percentage_curry=(count_makes_curry/(count_makes_curry+count_misses_curry))*100
print("Currys field goal pecentage is ",fgm_percentage_curry)
count_makes_klay=len(klay_data[klay_data['FGM']==1])
count_misses_klay=len(klay_data[klay_data['FGM']==0])
fgm_percentage_klay=(count_makes_klay/(count_makes_klay+count_misses_klay))*100
print("Klays field goal pecentage is ",fgm_percentage_klay)
if fgm_percentage_curry>fgm_percentage_klay:
    merits['Overall shooter:']=['Stephen Curry',fgm_percentage_curry,fgm_percentage_klay]
else:
    merits['Overall shooter:']=['Klay Thompson',fgm_percentage_curry,fgm_percentage_klay]
#print(merits)

#2pt shooter
curry_data_2pt=curry_data[curry_data['PTS_TYPE']==2]
count_makes_curry=len(curry_data_2pt[curry_data_2pt['FGM']==1])
count_misses_curry=len(curry_data_2pt[curry_data_2pt['FGM']==0])
fgm_percentage_curry=(count_makes_curry/(count_makes_curry+count_misses_curry))*100
print("Currys 2pt pecentage is ",fgm_percentage_curry)
klay_data_2pt=klay_data[klay_data['PTS_TYPE']==2]
count_makes_klay=len(klay_data_2pt[klay_data_2pt['FGM']==1])
count_misses_klay=len(klay_data_2pt[klay_data_2pt['FGM']==0])
fgm_percentage_klay=(count_makes_klay/(count_makes_klay+count_misses_klay))*100
print("Klays 2 pt pecentage is ",fgm_percentage_klay)
if fgm_percentage_curry>fgm_percentage_klay:
    merits['2pt shooter:']=['Stephen Curry',fgm_percentage_curry,fgm_percentage_klay]
else:
    merits['2pt shooter:']=['Klay Thompson',fgm_percentage_curry,fgm_percentage_klay]

#print(merits)

#3pt shooter
curry_data_3pt=curry_data[curry_data['PTS_TYPE']==3]
count_makes_curry=len(curry_data_3pt[curry_data_3pt['FGM']==1])
count_misses_curry=len(curry_data_3pt[curry_data_3pt['FGM']==0])
fgm_percentage_curry=(count_makes_curry/(count_makes_curry+count_misses_curry))*100
print("Currys 3pt pecentage is ",fgm_percentage_curry)
klay_data_3pt=klay_data[klay_data['PTS_TYPE']==3]
count_makes_klay=len(klay_data_3pt[klay_data_3pt['FGM']==1])
count_misses_klay=len(klay_data_3pt[klay_data_3pt['FGM']==0])
fgm_percentage_klay=(count_makes_klay/(count_makes_klay+count_misses_klay))*100
print("Klays 3 pt pecentage is ",fgm_percentage_klay)
if fgm_percentage_curry>fgm_percentage_klay:
    merits['3pt shooter:']=['Stephen Curry',fgm_percentage_curry,fgm_percentage_klay]
else:
    merits['3pt shooter:']=['Klay Thompson',fgm_percentage_curry,fgm_percentage_klay]

print(merits)

# early possesion
c_shotclk=curry_data['SHOT_CLOCK'].mean()
k_shotclk=klay_data['SHOT_CLOCK'].mean()
avg_shotclk=(c_shotclk+k_shotclk)/2
print(avg_shotclk)
curry_data_3pt=curry_data[curry_data['SHOT_CLOCK']>=avg_shotclk]
count_makes_curry=len(curry_data_3pt[curry_data_3pt['FGM']==1])
count_misses_curry=len(curry_data_3pt[curry_data_3pt['FGM']==0])
fgm_percentage_curry=(count_makes_curry/(count_makes_curry+count_misses_curry))*100
print("Currys early shot pecentage is ",fgm_percentage_curry)
klay_data_3pt=klay_data[klay_data['SHOT_CLOCK']>=avg_shotclk]
count_makes_klay=len(klay_data_3pt[klay_data_3pt['FGM']==1])
count_misses_klay=len(klay_data_3pt[klay_data_3pt['FGM']==0])
fgm_percentage_klay=(count_makes_klay/(count_makes_klay+count_misses_klay))*100
print("Klays early shot pecentage is ",fgm_percentage_klay)
if fgm_percentage_curry>fgm_percentage_klay:
    merits['early possesion shooter:']=['Stephen Curry',fgm_percentage_curry,fgm_percentage_klay]
else:
    merits['early possesion shooter:']=['Klay Thompson',fgm_percentage_curry,fgm_percentage_klay]

#late possesion shooter

curry_data_3pt=curry_data[curry_data['SHOT_CLOCK']<=(avg_shotclk)]
count_makes_curry=len(curry_data_3pt[curry_data_3pt['FGM']==1])
count_misses_curry=len(curry_data_3pt[curry_data_3pt['FGM']==0])
fgm_percentage_curry=(count_makes_curry/(count_makes_curry+count_misses_curry))*100
print("Currys late shot pecentage is ",fgm_percentage_curry)
klay_data_3pt=klay_data[klay_data['SHOT_CLOCK']<=(avg_shotclk)]
count_makes_klay=len(klay_data_3pt[klay_data_3pt['FGM']==1])
count_misses_klay=len(klay_data_3pt[klay_data_3pt['FGM']==0])
fgm_percentage_klay=(count_makes_klay/(count_makes_klay+count_misses_klay))*100
print("Klays late shot pecentage is ",fgm_percentage_klay)
if fgm_percentage_curry>fgm_percentage_klay:
    merits['late possesion shooter:']=['Stephen Curry',fgm_percentage_curry,fgm_percentage_klay]
else:
    merits['late possesion shooter:']=['Klay Thompson',fgm_percentage_curry,fgm_percentage_klay]



#min touch time shooter
c_touchtime=curry_data['TOUCH_TIME'].mean()
k_touchtime=klay_data['TOUCH_TIME'].mean()
avg_touchtime=(c_touchtime+k_touchtime)/2
#print(avg_touchtime)

curry_data_3pt=curry_data[curry_data['TOUCH_TIME']<=avg_touchtime]
count_makes_curry=len(curry_data_3pt[curry_data_3pt['FGM']==1])
count_misses_curry=len(curry_data_3pt[curry_data_3pt['FGM']==0])
fgm_percentage_curry=(count_makes_curry/(count_makes_curry+count_misses_curry))*100
print("Currys percentage with less than average touch time ",fgm_percentage_curry)
klay_data_3pt=klay_data[klay_data['TOUCH_TIME']<=avg_touchtime]
count_makes_klay=len(klay_data_3pt[klay_data_3pt['FGM']==1])
count_misses_klay=len(klay_data_3pt[klay_data_3pt['FGM']==0])
fgm_percentage_klay=(count_makes_klay/(count_makes_klay+count_misses_klay))*100
print("Klays percentage with less than average touch time ",fgm_percentage_klay)
if fgm_percentage_curry>fgm_percentage_klay:
    merits['shooter with less touch time:']=['Stephen Curry',fgm_percentage_curry,fgm_percentage_klay]
else:
    merits['shooter with less touch time:']=['Klay Thompson',fgm_percentage_curry,fgm_percentage_klay]


#lesser dribbles

c_touchtime=curry_data['DRIBBLES'].mean()
k_touchtime=klay_data['DRIBBLES'].mean()
avg_touchtime=(c_touchtime+k_touchtime)/2
print(avg_touchtime)
curry_data_3pt=curry_data[curry_data['DRIBBLES']<=avg_touchtime]
count_makes_curry=len(curry_data_3pt[curry_data_3pt['FGM']==1])
count_misses_curry=len(curry_data_3pt[curry_data_3pt['FGM']==0])
fgm_percentage_curry=(count_makes_curry/(count_makes_curry+count_misses_curry))*100
print("Currys percentage with less than average dribbles ",fgm_percentage_curry)
klay_data_3pt=klay_data[klay_data['DRIBBLES']<=avg_touchtime]
count_makes_klay=len(klay_data_3pt[klay_data_3pt['FGM']==1])
count_misses_klay=len(klay_data_3pt[klay_data_3pt['FGM']==0])
fgm_percentage_klay=(count_makes_klay/(count_makes_klay+count_misses_klay))*100
print("Klays percentage with less than average dribbles ",fgm_percentage_klay)
if fgm_percentage_curry>fgm_percentage_klay:
    merits['shooter with less dribbles:']=['Stephen Curry',fgm_percentage_curry,fgm_percentage_klay]
else:
    merits['shooter with less dribbles:']=['Klay Thompson',fgm_percentage_curry,fgm_percentage_klay]



#early form
c_touchtime=curry_data['SHOT_NUMBER'].mean()
k_touchtime=klay_data['SHOT_NUMBER'].mean()
avg_touchtime=(c_touchtime+k_touchtime)/2
print(avg_touchtime)
curry_data_3pt=curry_data[curry_data['SHOT_NUMBER']<=avg_touchtime]
count_makes_curry=len(curry_data_3pt[curry_data_3pt['FGM']==1])
count_misses_curry=len(curry_data_3pt[curry_data_3pt['FGM']==0])
fgm_percentage_curry=(count_makes_curry/(count_makes_curry+count_misses_curry))*100
print("Currys percentage with less than average number of shots ",fgm_percentage_curry)
klay_data_3pt=klay_data[klay_data['SHOT_NUMBER']<=avg_touchtime]
count_makes_klay=len(klay_data_3pt[klay_data_3pt['FGM']==1])
count_misses_klay=len(klay_data_3pt[klay_data_3pt['FGM']==0])
fgm_percentage_klay=(count_makes_klay/(count_makes_klay+count_misses_klay))*100
print("Klays percentage with less than average number of shots ",fgm_percentage_klay)
if fgm_percentage_curry>fgm_percentage_klay:
    merits['shooter with less shots:']=['Stephen Curry',fgm_percentage_curry,fgm_percentage_klay]
else:
    merits['shooter with less shots:']=['Klay Thompson',fgm_percentage_curry,fgm_percentage_klay]

#closest defender

c_touchtime=curry_data['CLOSE_DEF_DIST'].mean()
k_touchtime=klay_data['CLOSE_DEF_DIST'].mean()
avg_touchtime=(c_touchtime+k_touchtime)/2

curry_data_3pt=curry_data[curry_data['CLOSE_DEF_DIST']<=avg_touchtime]
count_makes_curry=len(curry_data_3pt[curry_data_3pt['FGM']==1])
count_misses_curry=len(curry_data_3pt[curry_data_3pt['FGM']==0])
fgm_percentage_curry=(count_makes_curry/(count_makes_curry+count_misses_curry))*100
print("Currys percentage while guarded closely ",fgm_percentage_curry)
klay_data_3pt=klay_data[klay_data['CLOSE_DEF_DIST']<=avg_touchtime]
count_makes_klay=len(klay_data_3pt[klay_data_3pt['FGM']==1])
count_misses_klay=len(klay_data_3pt[klay_data_3pt['FGM']==0])
fgm_percentage_klay=(count_makes_klay/(count_makes_klay+count_misses_klay))*100
print("Klays percentage while guarded closely ",fgm_percentage_klay)
if fgm_percentage_curry>fgm_percentage_klay:
    merits['shot maker with tight defense:']=['Stephen Curry',fgm_percentage_curry,fgm_percentage_klay]
else:
    merits['shotmaker with tight defense:']=['Klay Thompson',fgm_percentage_curry,fgm_percentage_klay]
print(merits)


#first quater shooter
curry_data_3pt=curry_data[curry_data['PERIOD']==1]
count_makes_curry=len(curry_data_3pt[curry_data_3pt['FGM']==1])
count_misses_curry=len(curry_data_3pt[curry_data_3pt['FGM']==0])
fgm_percentage_curry=(count_makes_curry/(count_makes_curry+count_misses_curry))*100
print("Currys fgm percentage in first quater ",fgm_percentage_curry)
klay_data_3pt=klay_data[klay_data['PERIOD']==1]
count_makes_klay=len(klay_data_3pt[klay_data_3pt['FGM']==1])
count_misses_klay=len(klay_data_3pt[klay_data_3pt['FGM']==0])
fgm_percentage_klay=(count_makes_klay/(count_makes_klay+count_misses_klay))*100
print("Klays fgm percentage in first quater ",fgm_percentage_klay)
if fgm_percentage_curry>fgm_percentage_klay:
    merits['1st quater shooter:']=['Stephen Curry',fgm_percentage_curry,fgm_percentage_klay]
else:
    merits['1st quater shooter:']=['Klay Thompson',fgm_percentage_curry,fgm_percentage_klay]

# second quater shooter
curry_data_3pt=curry_data[curry_data['PERIOD']==2]
count_makes_curry=len(curry_data_3pt[curry_data_3pt['FGM']==1])
count_misses_curry=len(curry_data_3pt[curry_data_3pt['FGM']==0])
fgm_percentage_curry=(count_makes_curry/(count_makes_curry+count_misses_curry))*100
print("Currys fgm percentage in second quater ",fgm_percentage_curry)
klay_data_3pt=klay_data[klay_data['PERIOD']==2]
count_makes_klay=len(klay_data_3pt[klay_data_3pt['FGM']==1])
count_misses_klay=len(klay_data_3pt[klay_data_3pt['FGM']==0])
fgm_percentage_klay=(count_makes_klay/(count_makes_klay+count_misses_klay))*100
print("Klays fgm percentage in second quater ",fgm_percentage_klay)
if fgm_percentage_curry>fgm_percentage_klay:
    merits['2nd quater shooter:']=['Stephen Curry',fgm_percentage_curry,fgm_percentage_klay]
else:
    merits['2nd quater shooter:']=['Klay Thompson',fgm_percentage_curry,fgm_percentage_klay]


#3rd quater shooter
curry_data_3pt=curry_data[curry_data['PERIOD']==3]
count_makes_curry=len(curry_data_3pt[curry_data_3pt['FGM']==1])
count_misses_curry=len(curry_data_3pt[curry_data_3pt['FGM']==0])
fgm_percentage_curry=(count_makes_curry/(count_makes_curry+count_misses_curry))*100
print("Currys fgm percentage in third quater ",fgm_percentage_curry)
klay_data_3pt=klay_data[klay_data['PERIOD']==3]
count_makes_klay=len(klay_data_3pt[klay_data_3pt['FGM']==1])
count_misses_klay=len(klay_data_3pt[klay_data_3pt['FGM']==0])
fgm_percentage_klay=(count_makes_klay/(count_makes_klay+count_misses_klay))*100
print("Klays fgm percentage in third quater ",fgm_percentage_klay)
if fgm_percentage_curry>fgm_percentage_klay:
    merits['3rd quater shooter:']=['Stephen Curry',fgm_percentage_curry,fgm_percentage_klay]
else:
    merits['3rd quater shooter:']=['Klay Thompson',fgm_percentage_curry,fgm_percentage_klay]


#4th quater shooting
curry_data_3pt=curry_data[curry_data['PERIOD']==4]
count_makes_curry=len(curry_data_3pt[curry_data_3pt['FGM']==1])
count_misses_curry=len(curry_data_3pt[curry_data_3pt['FGM']==0])
fgm_percentage_curry=(count_makes_curry/(count_makes_curry+count_misses_curry))*100
print("Currys fgm percentage in fourth quater ",fgm_percentage_curry)
klay_data_3pt=klay_data[klay_data['PERIOD']==4]
count_makes_klay=len(klay_data_3pt[klay_data_3pt['FGM']==1])
count_misses_klay=len(klay_data_3pt[klay_data_3pt['FGM']==0])
fgm_percentage_klay=(count_makes_klay/(count_makes_klay+count_misses_klay))*100
print("Klays fgm percentage in fourth quater ",fgm_percentage_klay)
if fgm_percentage_curry>fgm_percentage_klay:
    merits['4th quater shooter:']=['Stephen Curry',fgm_percentage_curry,fgm_percentage_klay]
else:
    merits['4th quater shooter:']=['Klay Thompson',fgm_percentage_curry,fgm_percentage_klay]

print('category:                                 best choice')
for name, age in merits.items():
    print('{} {}'.format(name, age))

data={'categories':['Overall','2pt','3pt','early possesion','late possesion','less touch time','tight defense','1st quater','2nd quater','3rd quater','4th quater','less dribbles','shooter with less shots']
,'Best choice':[merits['Overall shooter:'][0],merits['2pt shooter:'][0],merits['3pt shooter:'][0],merits['early possesion shooter:'][0],merits['late possesion shooter:'][0],merits['shooter with less touch time:'][0],merits['shot maker with tight defense:'][0],merits['1st quater shooter:'][0],merits['2nd quater shooter:'][0],merits['3rd quater shooter:'][0],merits['4th quater shooter:'][0],merits['shooter with less dribbles:'][0],merits['shooter with less shots:'][0]],'Currys percentage':[merits['Overall shooter:'][1],merits['2pt shooter:'][1],merits['3pt shooter:'][1],merits['early possesion shooter:'][1],merits['late possesion shooter:'][1],merits['shooter with less touch time:'][1],merits['shot maker with tight defense:'][1],merits['1st quater shooter:'][1],merits['2nd quater shooter:'][1],merits['3rd quater shooter:'][1],merits['4th quater shooter:'][1],merits['shooter with less dribbles:'][1],merits['shooter with less shots:'][1]],'Klays percentage':[merits['Overall shooter:'][2],merits['2pt shooter:'][2],merits['3pt shooter:'][2],merits['early possesion shooter:'][2],merits['late possesion shooter:'][2],merits['shooter with less touch time:'][2],merits['shot maker with tight defense:'][2],merits['1st quater shooter:'][2],merits['2nd quater shooter:'][2],merits['3rd quater shooter:'][2],merits['4th quater shooter:'][2],merits['shooter with less dribbles:'][2],merits['shooter with less shots:'][2]],}
#print(data)
df = pd.DataFrame(data)
 
# Print the output.
print(df)

#ax = df[['Currys percentage','Klays percentage']].plot(kind='bar', title ="overall analysis", figsize=(15, 10), legend=True, fontsize=12)
ax = df.plot.bar(x='categories', y=['Currys percentage','Klays percentage'], rot=0,fontsize=8)

plt.show()
    







#purple is miss yellow made
plt.scatter(curry_data['SHOT_DIST'],curry_data['CLOSE_DEF_DIST'],c=curry_data['FGM'])
plt.title('field goal WRT shot distance and closest defender(Curry)')
#plt.annotate('Make', xy=(2.5, 29), xytext=(14, 28),
#           arrowprops=dict(facecolor='black', shrink=0.05))
plt.text(25,29, 'Yellow->make,purple ->miss', style='italic')
plt.xlabel('Shot distance')
plt.ylabel('Closest Defender distance')
plt.show()

plt.scatter(curry_data['SHOT_DIST'],curry_data['TOUCH_TIME'],c=curry_data['FGM'])
plt.title('field goal WRT shot distance and Touch time(Curry)')
plt.text(25,20, 'Yellow->make,purple ->miss', style='italic')
plt.xlabel('Shot distance')
plt.ylabel('Touch time')
plt.show()

plt.scatter(klay_data['SHOT_DIST'],klay_data['CLOSE_DEF_DIST'],c=klay_data['FGM'])
plt.title('field goal WRT shot distance and closest defender(Klay)')
plt.text(25,40, 'Yellow->make,purple ->miss', style='italic')
plt.xlabel('Shot distance')
plt.ylabel('Closest Defender distance')
plt.show()

plt.scatter(klay_data['SHOT_DIST'],klay_data['TOUCH_TIME'],c=klay_data['FGM'])
plt.title('field goal WRT shot distance and Touch time(Klay)')
plt.text(35,11, 'Yellow->make,purple ->miss', style='italic')
plt.xlabel('Shot distance')
plt.ylabel('Touch time')
plt.show()



#X=train[['SHOT_NUMBER','PERIOD','SHOT_CLOCK','DRIBBLES','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST']] shot clocl is problematic line
X=train[['SHOT_NUMBER','PERIOD','DRIBBLES','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST']]
Y=train['FGM']
print(X.shape)
print(Y.shape)
print(train['FGM'].value_counts())
count_makes=len(train[train['FGM']==1])
count_misses=len(train[train['FGM']==0])
fgm_percentage=(count_makes/(count_makes+count_misses))*100
print("combined field goal pecentage is ",fgm_percentage)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split( X, Y, test_size = 0.25, random_state = 0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)
print ("Confusion Matrix : \n", cm)
from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(ytest, y_pred))
