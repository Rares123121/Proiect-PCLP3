import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
gender_submission=pd.read_csv('gender_submission.csv')

def initial_g(df,col,nume_file):
    for i in col:
        plt.figure(figsize=(10,6))
        sns.histplot(df[i].dropna(),kde=True)
        plt.title(f'Distribuția variabilei {i}')
        plt.savefig(f'{nume_file}_{i}.png')
        plt.close()

coloane=['Age','Fare']
initial_g(train, coloane,'initial')

#task 1
def eliminare_outliers_iqr(df,col):
    for i in col:
        q1=df[i].quantile(0.25)
        q3=df[i].quantile(0.75)
        iqr=q3 - q1
        jos=q1 - 1.5 * iqr
        sus=q3 + 1.5 * iqr
        df = df[(df[i] >= jos) & (df[i] <= sus)]
    return df

curatat=eliminare_outliers_iqr(train.copy(),coloane)
initial_g(curatat,coloane,'curatat')

curatat.to_csv('modificat.csv',index=False)

#Task 2
def eliminare_outliers_zscore(df,col,eroare=3):
    indice=np.ones(len(df),dtype=bool)
    for i in col:
        col_scor=zscore(df[i].dropna())
        zscor=np.abs(col_scor)
        valid=(zscor<eroare).reindex(df[i].index,fill_value=False)
        indice &= valid
    return df[indice]

zscore_curatat=eliminare_outliers_zscore(train.copy(),coloane)
initial_g(zscore_curatat,coloane,'zscore_curatat')

#Task 3
print("La IQR am calculat Q1, Q3 si IQR, iar valorie care nu erau in intervalul dat au fost scoase")
print("La Z-score, am caluculat z-score pentru fiecare valoare, iar cele care erau mai mari in modul decat 3 au fost scoase")

#Task 4
def data(df):
    df=df.drop(columns=['Name','Ticket','Cabin'])
    
    imp=SimpleImputer(strategy='mean')
    df['Age']=imp.fit_transform(df[['Age']])
    df['Fare']=imp.fit_transform(df[['Fare']])
    
    encode=LabelEncoder()
    df['Sex']=encode.fit_transform(df['Sex'])
    df['Embarked']=encode.fit_transform(df['Embarked'].fillna('S'))
    
    scalar=StandardScaler()
    df[['Age','Fare']]=scalar.fit_transform(df[['Age','Fare']])
    return df

traind, vald=train_test_split(curatat,test_size=0.2,random_state=42)
traind=data(traind)
vald=data(vald)

x_train=traind.drop(columns=['Survived'])
y_train=traind['Survived']
x=vald.drop(columns=['Survived'])
y=vald['Survived']

model=RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
y_predicite=model.predict(x)

acuratete = accuracy_score(y, y_predicite)
print(f'Acuratetea este: {acuratete}')
print('erori:')
print(confusion_matrix(y, y_predicite))

plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y, y_predicite), annot=True, fmt='d', cmap='Blues')
plt.title('Erori')
plt.savefig('erori.png')
plt.close()