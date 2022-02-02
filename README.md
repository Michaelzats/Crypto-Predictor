import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set()
from sklearn.preprocessing import LabelEncoder
import statistics
from random import randint 
from collections import Counter 



# DATA CLEANNING #1

df = pd.read_csv('answers.csv')
df = df.replace(',','', regex=True)


df2 = df.rename({'Отметка времени': 'Time', 
                'Have you ever invested money in the following (select all applied) ':'Invested_in',
                'If you have invested, how long have you been investing? (Answer if you invested before)': 'Duration_investment',
                'If you have invested, how much money have you invested? (Answer if you answered yes before)': 'Amount_invested',
                'What do you think the best investment strategy is? ': 'Best_method',
                'How frequently do you follow the financial news?': 'News_frequency',
                'What is the Purpose of the Investing? (Select all applied to you)': 'Investment_goal',
                'Do you know these market crashes? (Select all you know)': 'Crashes',
                'Do you know what the Technical and Fundamental Analyses are?': 'Analyses',
                'How much risk are you willing to take in the investments? (1 not at all, 5 much risk)': 'Investment_risk_willingness',
                'How emotional are you? (1 not at all, 5 super emotional)': 'Emotional_stability',
                'Do emotions play any role in your daily life?(1 not at all, 5 much)': 'Role_emotions',
                'Have you ever made any financial decisions because of emotions?': 'Emotional_decisions',
                'Do you trust your gut feeling with financial decisions?': 'Gut_feeling',
                'Where do you come from?': 'Country',
                'What gender are you?': 'Gender',
                'How old are you?':'Age',
                'What is your highest level of education? ': 'Education'}, axis=1)

df = df2.rename({'Have you ever invested money in the following (select all applied)':'Invested_in'})
df.head()

df_nonone = df.dropna()     
print (len(df_nonone))


df_Age.replace({"40+":"41+"}, inplace=True);



df_invested_in = df['Invested_in']
df_Duration_investment = df['Duration_investment']
df_Duration_investment_experement = df['Duration_investment']
df_Amount_invested = df['Amount_invested']
df_Best_method = df['Best_method']
df_News_frequency = df['News_frequency']
df_Investment_goal = df['Investment_goal']
df_Crashes = df['Crashes']
df_Analyses = df['Analyses']
df_Investment_risk_willingness = df['Investment_risk_willingness']
df_Emotional_stability = df['Emotional_stability']
df_Role_emotions = df['Role_emotions']
df_Emotional_decisions = df['Emotional_decisions']
df_Gut_feeling = df['Gut_feeling']
df_Country = df['Country']
df_Gender = df['Gender']
df_Role_emotions = df['Role_emotions']
df_Age = df['Age']
df_Education = df['Education']


# ANALYSES GENERAL #2

# Invested_in ANALYTICS


num_invested = df.groupby("Invested_in").size()
num_invested


max(num_invested)
# Stocks

min(num_invested)
# Crypto I have never invested                     

q= df.Invested_in.str.count("Crypto").sum()
q

w=df.Invested_in.str.count("I have never invested").sum()
w

e=df.Invested_in.str.count("Materials").sum()
e


r=df.Invested_in.str.count("Property").sum()
r


t=df.Invested_in.str.count("Stocks").sum()
t



types_of_investments = ['Stocks','Crypto','Property','Materials','Never invested']
occurance = [t,q,r,e,w]
# occurance.sort()

New_Colors = ['green','blue','purple','brown','teal']
plt.bar(types_of_investments, occurance,color=New_Colors)
plt.title('Occurrence of Investments Methods Answers from Russians', fontsize=14)
plt.xlabel('Types of Investments', fontsize=14)
plt.ylabel('Occurrence', fontsize=14)
plt.grid(True)

for index,data in enumerate(occurance):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12))
plt.tight_layout()

plt.show()
print (len(df.Invested_in))


# duration_invested ANALYTICS


duration_invested = df.groupby("Duration_investment").size()
za = 89
xa = 122
ca = 68
va = 72
duration_invested

Years_of_Investments = ['>1','1:2','2:4','<=5']
occurance = [ca,za,xa,va,]
# occurance.sort()

New_Colors = ['green','blue','purple','brown','teal']
plt.bar(Years_of_Investments, occurance,color=New_Colors)
plt.title('How Long Russians Invest', fontsize=14)
plt.xlabel('Years of Investments', fontsize=14)
plt.ylabel('Occurrence', fontsize=14)
plt.grid(True)

for index,data in enumerate(occurance):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12))
plt.tight_layout()

plt.show()
sum(occurance)

less_one_years = [1]
statistics.mean(less_one_years)

one_two_years = [1,2]
statistics.mean(one_two_years)



from_two_thorg_years = [2,4]
statistics.mean(from_two_thorg_years)


more_than_five = [5]
statistics.mean(more_than_five)



# # # # Standard version of describe statistics

df_Duration_investment.replace({"From 1 to 2 years": 1.5, 
                                   "Less than 1 year": 1,
                                   "More than 5 years":5,
                                  "From 2 to 4 years":3}, inplace=True);

df_Duration_investment.describe()



# # Generating the random Probability for the statistics from one year

# from random import randint 
# from collections import Counter 
 
# rolls = [randint(0,12) for _ in range(68)] 

 
# print(Counter(rolls)) 



# Randomly Gnerated number of months invested with 68 participants
((11*9)+(10*8)+(12*8)+(6*7)+(5*6)+(2*6)+(3*5)+(8*5)+(4*4)+(7*4)+(9*3)+(1*2)+(1*0))/68
# It is 0.583333 = 0.6 of the year

# from random import randint from 5 to 10 years investings
# from collections import Counter 
 
# rolls = [randint(5,10) for _ in range(72)] 
 
# print(Counter(rolls)) 

# Randomly Gnerated year from 5 to 10 invested with 72 participants

((6*17)+(9*15)+(8*13)+(10*10)+(5*9)+(7*8))/72



# Random Mean alterantive of years 
((68*0.7)+(89*1.5)+(122*3)+(72*7.5))/351

# amount_invested ANALYTICS


Amount_invested = df.groupby("Amount_invested").size()
zz = 53
xz = 92
cz = 73
vz = 34
bz = 50
nz = 53
Amount_invested

amount_of_investments = ['>1','1:3','3:7','7:20','<20','Not to say']
occurance = [vz,zz,xz,cz,bz,nz]
# occurance.sort()

New_Colors = ['green','blue','purple','brown','teal']
plt.bar(amount_of_investments, occurance,color=New_Colors)
plt.title('Amount of Russians Investments', fontsize=14)
plt.xlabel('Amount of Investments(in € Thousands)', fontsize=14)
plt.ylabel('Occurrence', fontsize=14)
plt.grid(True)

for index,data in enumerate(occurance):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12))
plt.tight_layout()

plt.show()
df.Amount_invested.describe()

less_than_thousand = [1]
statistics.mean(less_one_years)

from_thousand_to_three = [1,3]
statistics.mean(from_thousand_to_three)

from_three_to_seven = [3,7]
statistics.mean(from_three_to_seven)

from_seven_to_twenty = [7,20]
statistics.mean(from_seven_to_twenty)

from_twenty = [20]
statistics.mean(from_twenty)

# # # # Standard version of describe statistics

df_Amount_invested.replace({"From 1000 EUR to 3000 EUR":2, 
                                   "Less than 1000 EUR":1,
                                   "From 7000 EUR to 20000 EUR":13.5,
                                  "More than 20000 EUR":20,
                                  "From 3000 EUR to 7000 EUR":5,
                                  "Prefer not to say": None}, inplace=True);

df_Amount_invested.describe()

# # Generating the random Probability for the statistics from 1000 and less euro

# from random import randint 
# from collections import Counter 
 
# rolls = [randint(1,10) for _ in range(34)] 

 
# print(Counter(rolls)) 


# Generating the random Probability for the statistics from 1000 and less euro
((8*6)+(5*5)+(9*4)+(4*4)+(6*4)+(7*3)+(3*3)+(2*3)+(10*2)/31)

# # Generating the random Probability for the statistics from 20.000 and more euro

# from random import randint 
# from collections import Counter 
 
# rolls = [randint(20,100) for _ in range(50)] 

 
print(Counter(rolls)) 

# Generating the random Probability for the statistics from 1000 and less euro
((57*4)+(68*4)+(26*2)+(72*2)+(43*2)+(29*2)+(49*2)+(28*2)+(87*2)+(41*1)+(32*1)+(50*1)+(30*1)+(44*1)+(79*1)+(7*1)+(99*1)+(80*1)+(74*1)+(39*1)+(24*1)+(67*1)+(89*1)+(91*1)+(47*1)+(36*1)+(83*1)+(40*1)+(78*1)+(34*1)+(31*1)+(84*1)+(98*1)+(77*1)+(27*1)+(65*1)+(64*1)+(33*1)/50)

# Random Mean alterantive of invesments 
((34*0.185)+(53*2)+(92*5)+(73*13.5)+(50*27.8))/302

# best_investment ANALYTICS


best_invested = df.groupby("Best_method").size()
zq = 25
xq = 78
cq = 125
vq = 24
bq = 35
nq = 88
best_invested




Best_invested = ['Bank Deposit','Other','Property','Crypto','Stocks','Different Assets']
occurance = [vq,zq,xq,cq,bq,nq]
occurance.sort()

New_Colors = ['green','blue','purple','brown','teal']
plt.bar(Best_invested, occurance,color=New_Colors)
plt.title('Best Methods of Investments by Russians', fontsize=14)
plt.xlabel('Types of Investments', fontsize=14)
plt.ylabel('Occurrence', fontsize=14)
plt.grid(True)

for index,data in enumerate(occurance):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12))
plt.tight_layout()

plt.show()
sum(occurance)

# News frequency ANALYTICS


news_frequency = df.groupby("News_frequency").size()
zqq = 47
xqq = 127
cqq = 113
vqq = 88

news_frequency



# news_frequency = ['Never','Rarely','Often','Really often']
# occurance = [zqq,cqq,xqq,vqq]
# # occurance.sort()

# New_Colors = ['green','blue','purple','brown','teal']
# plt.bar(news_frequency, occurance,color=New_Colors)
# plt.title('Frequency of Financial-news followed by Russians', fontsize=14)
# plt.xlabel('Frequency', fontsize=14)
# plt.ylabel('Occurrence', fontsize=14)
# plt.grid(True)

# for index,data in enumerate(occurance):
#     plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12))
# plt.tight_layout()

# plt.show()
# sum(occurance)

df.News_frequency.describe()

# Goals of investmesnts ANALYTICS


investment_goal = df.groupby("Investment_goal").size()
investment_goal

max(investment_goal)
# Other reasons


min(investment_goal)
# For the retirement For buying a car For family and children


aa= df.Investment_goal.str.count("For the future safety").sum()
aa


bb = df.Investment_goal.str.count("For becoming rich").sum()
bb

cc = df.Investment_goal.str.count("For buying a house").sum()
cc


vv = df.Investment_goal.str.count("For the retirement").sum()
vv



nn = df.Investment_goal.str.count("For buying a car").sum()
nn


mm = df.Investment_goal.str.count("For College/University").sum()
mm


ww = df.Investment_goal.str.count("For family and children").sum()
ww


ee = df.Investment_goal.str.count("Other reasons").sum()
ee


Investment__goal = ['For the future safety','For becoming rich','For the retirement','For family and children','For buying a house','For buying a car','Other reasons','For College/University']
occurance = [aa,bb,cc,vv,nn,mm,ww,ee]
# occurance.sort()

plt.figure(figsize=(10,8))


New_Colors = ['green','blue','purple','brown','teal','black','orange']
plt.bar(Investment__goal, occurance,color=New_Colors)
plt.title('Goals of Investing by Russians', fontsize=14)
plt.xlabel('Goals', fontsize=14)
plt.ylabel('Occurrence', fontsize=14)
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large')
plt.grid(True)


for index,data in enumerate(occurance):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12))
plt.tight_layout()

plt.show()
print (len(df.Investment_goal))


# Crashes_awareness ANALYTICS


Crashes_awareness = df.groupby("Crashes").size()
Crashes_awareness

max(Crashes_awareness)
# Financial crisis of 2007–08 Cryptocurrency crash of 2018 March Covid-19 crash of 2020                                                        


min(Crashes_awareness)
# many


qqq= df.Crashes.str.count("Wall Street Crash of 1929").sum()
www= df.Crashes.str.count("Russian financial crisis of 1998").sum()
eee= df.Crashes.str.count("Dot-com bubble of 2000").sum()
rrr= df.Crashes.str.count("Financial crisis of 2007–08").sum()
ttt= df.Crashes.str.count("Cryptocurrency crash of 2018").sum()
yyy= df.Crashes.str.count("Chinese stock bubble of 2007").sum()
uuu= df.Crashes.str.count("March Covid-19 crash of 2020").sum()
iii= df.Crashes.str.count("Other").sum()
ooo= df.Crashes.str.count("I do not know any").sum()












Investment__goal = ['Wall Street Crash of 1929','Russian financial crisis of 1998','Dot-com bubble of 2000','Financial crisis of 2007–08','Cryptocurrency crash of 2018','Chinese stock bubble of 2007','March Covid-19 crash of 2020','Other','I do not know any']
occurance = [qqq,www,eee,rrr,ttt,yyy,uuu,iii,ooo]
# occurance.sort()

plt.figure(figsize=(10,8))


New_Colors = ['green','blue','purple','brown','teal','black','orange']
plt.bar(Investment__goal, occurance,color=New_Colors)
plt.title('Known Financial Crashes by Russians ', fontsize=14)
plt.xlabel('Crashes', fontsize=14)
plt.ylabel('Occurrence', fontsize=14)
plt.grid(True)
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large')

for index,data in enumerate(occurance):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12))
plt.tight_layout()

plt.show()
print (len(df.Crashes))


# Analyses ANALYTICS


Analyses_knowing = df.groupby("Analyses").size()
qqqqq = 127
wwwww = 65
eeeee = 73
rrrrr = 114
Analyses_knowing

analyses_knowing = ['No at all','Both','Technical','Fundamental']
occurance = [qqqqq,tttt,eeeee,wwwww]
# occurance.sort()

# plt.figure(figsize=(10,8))


New_Colors = ['green','blue','purple','brown','teal','black','orange']
plt.bar(analyses_knowing, occurance,color=New_Colors)
plt.title('Familiarity of Russians with Technical and Fundamental Analyses', fontsize=14)
plt.xlabel('Type known', fontsize=14)
plt.ylabel('Occurrence', fontsize=14)
plt.grid(True)
# plt.xticks(
#     rotation=45, 
#     horizontalalignment='right',
#     fontweight='light',
#     fontsize='x-large')

for index,data in enumerate(occurance):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12))
plt.tight_layout()

plt.show()
sum(occurance)



# Risk_willingness_investments ANALYTICS


Risk_willingness_investments = df.groupby("Investment_risk_willingness").size()
qqqq = 50
wwww = 86
eeee = 112
rrrr = 87
tttt = 42
Risk_willingness_investments


Risk_willingness_investments = ['1(not willing)','2','3','4','5(Maximum willing)']
occurance = [qqqq,wwww,eeee,rrrr,tttt]
# occurance.sort()

# plt.figure(figsize=(10,8))


New_Colors = ['green','blue','purple','brown','teal','black','orange']
plt.bar(Risk_willingness_investments, occurance,color=New_Colors)
plt.title('Willingness to Take Financial Risk by Russians', fontsize=14)
plt.xlabel('Risk Levels', fontsize=14)
plt.ylabel('Occurrence', fontsize=14)
plt.grid(True)
# plt.xticks(
#     rotation=45, 
#     horizontalalignment='right',
#     fontweight='light',
#     fontsize='x-large')

for index,data in enumerate(occurance):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12))
plt.tight_layout()

plt.show()
sum(occurance)
df.Investment_risk_willingness.describe()

Risk_willingness_investments = df.groupby("Investment_risk_willingness").size()
qqqq = 50
wwww = 86
eeee = 112
rrrr = 87
tttt = 42
Risk_willingness_investments




# Emotional_stability ANALYTICS


emotional_stability = df.groupby("Emotional_stability").size()
qqqqqq = 40
wwwwww = 104
eeeeee = 148
rrrrrr = 58
tttttt = 25
emotional_stability 

emotional_stability = ['1(not emotional)','2','3','4','5(Maximum emotional)']
occurance = [qqqqqq,wwwwww,eeeeee,rrrrrr,tttttt]
# occurance.sort()

# plt.figure(figsize=(10,8))


New_Colors = ['green','blue','purple','brown','teal','black','orange']
plt.bar(emotional_stability, occurance,color=New_Colors)
plt.title('Emotional Stability from Russians', fontsize=14)
plt.xlabel('Emotional Stability Levels', fontsize=14)
plt.ylabel('Occurrence', fontsize=14)
plt.grid(True)
# plt.xticks(
#     rotation=45, 
#     horizontalalignment='right',
#     fontweight='light',
#     fontsize='x-large')

for index,data in enumerate(occurance):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12))
plt.tight_layout()

plt.show()
sum(occurance)
df.Emotional_stability.describe()

# Role_emotions ANALYTICS


role_emotions = df.groupby("Role_emotions").size()
qqqqqqqq = 52
wwwwwwww = 107
eeeeeeee = 140
rrrrrrrr = 57
tttttttt = 21
role_emotions 

role_emotions = ['1(not at all)','2','3','4','5(Extremly much)']
occurance = [qqqqqqqq,wwwwwwww,eeeeeeee,rrrrrrrr,tttttttt]
# occurance.sort()

# plt.figure(figsize=(10,8))


New_Colors = ['green','blue','purple','brown','teal','black','orange']
plt.bar(role_emotions, occurance,color=New_Colors)
plt.title('Importance of Emotions for Russians', fontsize=14)
plt.xlabel('Importance of Emotions', fontsize=14)
plt.ylabel('Occurrence', fontsize=14)
plt.grid(True)
# plt.xticks(
#     rotation=45, 
#     horizontalalignment='right',
#     fontweight='light',
#     fontsize='x-large')

for index,data in enumerate(occurance):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12))
plt.tight_layout()

plt.show()
sum(occurance)
df.Role_emotions.describe()

# Emotional_decisions ANALYTICS


emotional_decisions = df.groupby("Emotional_decisions").size()
yes = 230
no = 148
emotional_decisions  

emotional_decisions = ['yes','no']
occurance = [yes,no]
plt.pie(occurance, labels = emotional_decisions,autopct='%1.0f%%')
plt.title("Proportions of Russians Who Have Made Financial Decisions upon Emotions and Who Have not", bbox={'facecolor':'0.8', 'pad':5})
df.Emotional_decisions.describe()




# Gut_feeling ANALYTICS


gut_feeling = df.groupby("Gut_feeling").size()
yes1 = 228
no1 = 148
gut_feeling  

gut_feeling = ['yes','no']
occurance = [yes1,no1]
plt.pie(occurance, labels = gut_feeling,autopct='%1.0f%%')
plt.title("Proportions of Russians Who Trust the Gut-feeling with Financial Decisions and Who Do not", bbox={'facecolor':'0.8', 'pad':5})
df.Gut_feeling.describe()





# Country ANALYTICS


country = df.groupby("Country").size()
eu = 48
us = 27
ussr = 290
other = 21

country

gut_feeling = ['EU, Norway, Switzerland','Other Countries',"Post USSR Countries","US, CANADA" ]
occurance = [eu,other,ussr,us]
plt.pie(occurance, labels = gut_feeling,autopct='%1.0f%%')
plt.title("Proportions of Russians Who Trust the Gut-feeling with Financial Decisions and Who Do not", bbox={'facecolor':'0.8', 'pad':5})
df.Gut_feeling.describe()

# Gender ANALYTICS


gender = df.groupby("Gender").size()
Male = 235
Female = 121
Non_binary = 10
Other = 2
Prefer_not_to_say = 15

gender

gender_ = ['Female','Male','Non-binary','Other','Prefer not to say']
occurance = [Female,Male,Non_binary,Other,Prefer_not_to_say]
# occurance.sort()

# plt.figure(figsize=(10,8))


New_Colors = ['green','blue','purple','brown','teal','black','orange']
plt.bar(gender_, occurance,color=New_Colors)
plt.title('Genders of participants', fontsize=14)
plt.xlabel('Genders', fontsize=14)
plt.ylabel('Occurrence', fontsize=14)
plt.grid(True)
# plt.xticks(
#     rotation=45, 
#     horizontalalignment='right',
#     fontweight='light',
#     fontsize='x-large')

for index,data in enumerate(occurance):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12))
plt.tight_layout()

plt.show()
sum(occurance)
df.Gender.describe()

# gender = df.groupby("Gender").size()
# Male = 235
# Female = 121
# Non_binary = 10
# Other = 2
# Prefer_not_to_say = 15

# gender

indexNames = df[ df['Gender'] == 'Other' ].index
# Delete these row indexes from dataFrame
df.drop(indexNames , inplace=True)

indexNames = df[ df['Gender'] == 'Prefer not to say' ].index
# Delete these row indexes from dataFrame
df.drop(indexNames , inplace=True)

indexNames = df[ df['Gender'] == 'Non-binary' ].index
# Delete these row indexes from dataFrame
df.drop(indexNames , inplace=True)

df.Gender.unique()

# Age ANALYTICS

age = df.groupby("Age").size()
qw = 78
qe = 104
qr = 134
qt = 68

age

age = ['18-23','24-29','30-40','<41']
occurance = [qw,qe,qr,qt,]
# occurance.sort()

New_Colors = ['green','blue','purple','brown','teal']
plt.bar(age, occurance,color=New_Colors)
plt.title('Age Groups of Russians Investors', fontsize=14)
plt.xlabel('Age Groups', fontsize=14)
plt.ylabel('Occurrence', fontsize=14)
plt.grid(True)

for index,data in enumerate(occurance):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12))
plt.tight_layout()

plt.show()
df.Age.describe()

FIRST = [18,23]
SECOND = [24,29]
THERD = [30,40]
THOURH = [41]

statistics.mean(THERD)


df.Age.replace({"18-23": 20.5, 
                                   "24-29": 26.5,
                                   "30-40":35,
                                  "41+":41}, inplace=True);
df.Age.describe()

# Education ANALYTICS

education = df.groupby("Education").size()
Bachelor = 135
College = 40
Doctorate = 21
School = 37
Master = 89
Undergraduate = 61
education 

education = ['Bachelor','Community College Diploma','Doctorate','High School Diploma','Master','Undergraduate']
occurance = [Bachelor,College,Doctorate,School,Master,Undergraduate]
# occurance.sort()
plt.figure(figsize=(11,9))


New_Colors = ['green','blue','purple','brown','teal','orange']
plt.bar(education, occurance,color=New_Colors)
plt.title('Highest Education of Russians Investors', fontsize=14)
plt.xlabel('Education', fontsize=14)
plt.ylabel('Occurrence', fontsize=14)
plt.grid(True)

for index,data in enumerate(occurance):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12))
plt.tight_layout()

plt.show()
df.Education.describe()

df.head()

df.News_frequency.replace({"Never": 1, 
                                   "Rarely": 2,
                                   "Often":3,
                                  "Really often":4}, inplace=True);
df.Education.describe()
# df['Education'].value_counts(sort=True)

### ANALYSES COMPLICATED #3 
# education AND MONEY
plt.figure(figsize=(8,6))
plt.title('Correlation between Highest Education and Accumulated Investments from Russians Investors', fontsize=14)
sns.barplot(x="Amount_invested", y="Education", data=df)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Types of Education");



plt.figure(figsize=(16, 10))
plt.title('Correlation between Highest Education and Accumulated Investments from Russians Investors', fontsize=14)
sns.boxplot(x="Amount_invested", y="Education", data=df)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Types of Education");
plt.show()

plt.figure(figsize=(16, 10))
plt.title('Correlation between Highest Education and Accumulated Investments from Russians Investors', fontsize=14)
sns.violinplot(x="Amount_invested", y="Education", data=df)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Types of Education");
plt.show()

plt.figure(figsize=(16, 10))

plt.title('Correlation between Highest Education and Accumulated Investments with the Gender Comparison from Russians Investors', fontsize=14)
sns.violinplot(data=df, x="Amount_invested", y="Education",hue= 'Gender', split=True )
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Types of Education")    






# df.Education.describe()
df['Education'].value_counts(sort=True)
# CONFIRMED as it is less than 0.05
from scipy.stats import chi2_contingency

contingency1 = [[129, 67],
              [87, 49],
                [53, 51],
                [39, 85],
                [33, 32],
               [16,67]]

chi2_contingency(contingency1)

# df.Education.describe()
# df['Amount_invested'].value_counts(sort=True)

df['Education'].value_counts(sort=True)


# # Duration of Investing AND Education

plt.figure(figsize=(8,6))
plt.title('Correlation between Duration of Investing in Years and Highest Education from Russians Investors', fontsize=14)
sns.barplot(x="Duration_investment", y="Education", data=df)
plt.xlabel("Duration of Investing in Years")
plt.ylabel("Types of Education");

plt.figure(figsize=(16, 10))
plt.title('Correlation between Duration of Investing in Years and Highest Education from Russians Investors', fontsize=14)
sns.boxplot(x="Duration_investment", y="Education", data=df)
plt.xlabel("Duration of Investing in Years")
plt.ylabel("Types of Education");

plt.figure(figsize=(16, 10))
plt.title('Correlation between Duration of Investing in Years and Highest Education from Russians Investors', fontsize=14)
sns.violinplot(x="Duration_investment", y="Education", data=df)
plt.xlabel("Duration of Investing in Years")
plt.ylabel("Types of Education");

plt.figure(figsize=(16, 10))
plt.title('Correlation between Duration of Investing in Years and Highest Education with the Gender Comparison from Russians Investors', fontsize=14)
sns.violinplot(x="Duration_investment", y="Education", data=df, hue= 'Gender', split=True )
plt.xlabel("Duration if Investing in Years")
plt.ylabel("Types of Education");  



df.Education.describe()
df['Education'].value_counts(sort=True)
# CONFIRMED as it is less than 0.05
from scipy.stats import chi2_contingency

contingency1 = [[129, 114],
              [87, 69],
                [53, 62],
                [39, 83],
                [33, 83],
               [16,69]]
chi2_contingency(contingency1)

# df.Duration_investment.describe()
# df['Duration_investment'].value_counts(sort=True)

df['Education'].value_counts(sort=True)

# # Amount of Investing AND Duration



plt.figure(figsize=(8,6))
plt.title('Correlation between Duration of Investing in Years and Accumulated Investments from Russians Investors', fontsize=14)
sns.barplot(x="Amount_invested", y="Duration_investment", data=df)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Duration Of Investing in Years");



plt.figure(figsize=(16, 10))
plt.title('Correlation between Duration of Investing in Years and Accumulated Investments from Russians Investors', fontsize=14)
sns.boxplot(x="Amount_invested", y="Duration_investment", data=df)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Duration Of Investing in Years");
plt.show()

plt.figure(figsize=(16, 10))
plt.title('Correlation between Duration of Investing in Years and Accumulated Investments from Russians Investors', fontsize=14)
sns.violinplot(x="Amount_invested", y="Duration_investment", data=df)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Duration Of Investing in Years");
plt.show()

plt.figure(figsize=(16, 10))
plt.title('Correlation between Duration of Investing in Years and Accumulated Investments with the Gender Comparison from Russians Investors', fontsize=14)
sns.violinplot(data=df, x="Amount_invested", y="Duration_investment",hue= 'Gender', split=True )
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Duration Of Investing in Years") 

df.Education.describe()
df['Education'].value_counts(sort=True)
# CONFIRMED as it is less than 0.05
from scipy.stats import chi2_contingency

contingency1 = [[32, 62],
              [51, 83],
                [85, 114],
                [13.5, 114],
                [20, 69],]
chi2_contingency(contingency1)

# df.Duration_investment.describe()
# df['Amount_invested'].value_counts(sort=True)

df['Duration_investment'].value_counts(sort=True)

# Methods of Investing AND MONEY

plt.figure(figsize=(8,6))
sns.barplot(x="Amount_invested", y="Best_method", data=df)
plt.title('Correlation between Methods of Investments and Accumulated Investments from Russians Investors', fontsize=14)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Methods of Investments");



plt.figure(figsize=(16, 10))
sns.boxplot(x="Amount_invested", y="Best_method", data=df)
plt.title('Correlation between Methods of Investments and Accumulated Investments from Russians Investors', fontsize=14)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Methods of Investments");


plt.figure(figsize=(16, 10))
sns.violinplot(x="Amount_invested", y="Best_method", data=df)
plt.title('Correlation between Methods of Investments and Accumulated Investments from Russians Investors', fontsize=14)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Methods of Investments");


plt.figure(figsize=(16, 10))
sns.violinplot(data=df, x="Amount_invested", y="Best_method",hue= 'Gender', split=True )
plt.title('Correlation between Duration of Investing and Accumulated Investments with the Gender Comparison from Russians Investors', fontsize=14)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Duration Of Investing in Years") 


df.Education.describe()
df['Education'].value_counts(sort=True)
# CONFIRMED as it is less than 0.05
from scipy.stats import chi2_contingency

contingency1 = [[119, 67],
              [80, 85],
                [76, 51],
                [35, 49],
                [23, 67],
               [19,32]]
chi2_contingency(contingency1)

# df['Amount_invested'].value_counts(sort=True)

# # df.Duration_investment.describe()
df['Best_method'].value_counts(sort=True)

# # Education AND Newes Frequency


plt.figure(figsize=(8,6))
sns.barplot(x="News_frequency", y="Education", data=df)
plt.title('Correlation between Frequency of Following Financial News and Highest Education from Russians Investors', fontsize=14)
plt.xlabel("Frequency of Following Financial (1: Never, 4: Always)")
plt.ylabel("Type of Education");


plt.figure(figsize=(16, 10))
sns.boxplot(x="News_frequency", y="Education", data=df)
plt.title('Correlation between Frequency of Following Financial News and Highest Education from Russians Investors', fontsize=14)
plt.xlabel("Frequency of Following Financial (1: Never, 4: Always)")
plt.ylabel("Type of Education");





plt.figure(figsize=(16, 10))
sns.violinplot(x="News_frequency", y="Education", data=df)
plt.title('Correlation between Frequency of Following Financial News and Highest Education from Russians Investors', fontsize=14)
plt.xlabel("Frequency of Following Financial (1: Never, 4: Always)")
plt.ylabel("Type of Education");





plt.figure(figsize=(16, 10))
sns.violinplot(data=df, x="News_frequency", y="Education",hue= 'Gender', split=True )
plt.title('Correlation between Frequency of Following Financial News and Highest Education with the Gender Comparison from Russians Investors', fontsize=14)
plt.xlabel("Frequency of Following Financial (1: Never, 4: Always)")
plt.ylabel("Type of Education");


df.Education.describe()
df['Education'].value_counts(sort=True)
# CONFIRMED as it is less than 0.05
from scipy.stats import chi2_contingency

contingency1 = [[129, 120],
              [87, 120],
                [53, 109],
                [39, 109],
                [33, 40],
               [16,83]]
chi2_contingency(contingency1)

# df['Education'].value_counts(sort=True)



['News_frequency'].value_counts(sort=True)

# # Amount of Investing AND Newes Frequency NO CONNECTION


plt.figure(figsize=(8,6))
sns.barplot(x="Amount_invested", y="News_frequency", data=df)
plt.title('Correlation between Frequency of Following Financial News and Accumulated Investments from Russians Investors', fontsize=14)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Frequency of Following Financial (1: Never, 4: Always)");



plt.figure(figsize=(16, 10))
sns.boxplot(x="Amount_invested", y="News_frequency", data=df)
plt.title('Correlation between Frequency of Following Financial News and Accumulated Investments from Russians Investors', fontsize=14)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Frequency of Following Financial (1: Never, 4: Always)");




plt.figure(figsize=(16, 10))
sns.violinplot(x="Amount_invested", y="News_frequency", data=df)
plt.title('Correlation between Frequency of Following Financial News and Accumulated Investments from Russians Investors', fontsize=14)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Frequency of Following Financial (1: Never, 4: Always)");




plt.figure(figsize=(16, 10))
sns.violinplot(data=df, x="Amount_invested", y="News_frequency",hue= 'Gender', split=True )
plt.title('Correlation between Frequency of Following Financial News and Accumulated Investments with the Gender Comparison from Russians Investors', fontsize=14)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Frequency of Following Financial (1: Never, 4: Always)") 

df.Education.describe()
df['Education'].value_counts(sort=True)
# CONFIRMED as it is less than 0.05
from scipy.stats import chi2_contingency

contingency1 = [[85, 120],
              [67, 109],
                [51, 109],
                [49, 120],
                [32, 40]]
chi2_contingency(contingency1)

# df['Amount_invested'].value_counts(sort=True)




df['News_frequency'].value_counts(sort=True)



# Analyses AND MONEY NOT CONFIRMEND
plt.figure(figsize=(8,6))
plt.title('Correlation between Familiarity with Technical and Fundamental Analyses and Accumulated Investments from Russians Investors', fontsize=14)
sns.barplot(x="Amount_invested", y="Analyses", data=df)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Types of Knowing Analyses");



plt.figure(figsize=(16, 10))
sns.boxplot(x="Amount_invested", y="Analyses", data=df)
plt.title('Correlation between Familiarity with Technical and Fundamental Analyses and Accumulated Investments from Russians Investors', fontsize=14)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Types of Knowing Analyses");
plt.show()

plt.figure(figsize=(16, 10))
sns.violinplot(x="Amount_invested", y="Analyses", data=df)
plt.title('Correlation between Familiarity with Technical and Fundamental Analyses and Accumulated Investments from Russians Investors', fontsize=14)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Types of Knowing Analyses");
plt.show()

plt.figure(figsize=(16, 10))
sns.violinplot(data=df, x="Amount_invested", y="Analyses",hue= 'Gender', split=True )
plt.title('Correlation between Familiarity with Technical and Fundamental Analyses and Accumulated Investments with the Gender Comparison from Russians Investors', fontsize=14)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Types of Knowing Analyses")    







df.Education.describe()
df['Amount_invested'].value_counts(sort=True)
# CONFIRMED as it is less than 0.05
from scipy.stats import chi2_contingency

contingency1 = [[116, 51],
              [107, 49],
                [70, 85],
                [62, 67],
                [32, 116]]
chi2_contingency(contingency1)

# df['Amount_invested'].value_counts(sort=True)



df['Analyses'].value_counts(sort=True)



# # Amount of Investing AND Investment_risk_willingness Reject


plt.figure(figsize=(8,6))
sns.barplot(x="Amount_invested", y="Investment_risk_willingness", data=df)
plt.title('Correlation between Risk Willingness in Finance and Accumulated Investments from Russians Investors', fontsize=14)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");




plt.figure(figsize=(16, 10))
sns.boxplot(x="Amount_invested", y="Investment_risk_willingness", data=df)
plt.title('Correlation between Risk Willingness in Finance and Accumulated Investments from Russians Investors', fontsize=14)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");




plt.figure(figsize=(16, 10))
sns.violinplot(x="Amount_invested", y="Investment_risk_willingness", data=df)
plt.title('Correlation between Risk Willingness in Finance and Accumulated Investments from Russians Investors', fontsize=14)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");




plt.figure(figsize=(16, 10))
sns.violinplot(data=df, x="Amount_invested", y="Investment_risk_willingness",hue= 'Gender', split=True )
plt.title('Correlation between Risk Willingness in Finance and Accumulated Investments with the Gender Comparison from Russians Investors', fontsize=14)
plt.xlabel("Amount of Investments(in € Thousands)")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");



df.Education.describe()
df['Amount_invested'].value_counts(sort=True)
# CONFIRMED as it is less than 0.05
from scipy.stats import chi2_contingency

contingency1 = [[85, 109],
              [67, 83],
                [51, 80],
                [49, 109],
                [32, 34]]
chi2_contingency(contingency1)

# df['Amount_invested'].value_counts(sort=True)

df['Investment_risk_willingness'].value_counts(sort=True)



# # Education AND Investment_risk_willingness



plt.figure(figsize=(10,7))
sns.barplot(x="Education", y="Investment_risk_willingness", data=df)
plt.title('Correlation between Risk Willingness in Finance and Highest Education from Russians Investors', fontsize=14)
plt.xlabel("Type of Education")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");




plt.figure(figsize=(16, 10))
sns.boxplot(x="Education", y="Investment_risk_willingness", data=df)
plt.title('Correlation between Risk Willingness in Finance and Highest Education from Russians Investors', fontsize=14)
plt.xlabel("Type of Education")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");




plt.figure(figsize=(16, 10))
sns.violinplot(x="Education", y="Investment_risk_willingness", data=df)
plt.title('Correlation between Risk Willingness in Finance and Highest Education from Russians Investors', fontsize=14)
plt.xlabel("Type of Education")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");





plt.figure(figsize=(16, 10))
sns.violinplot(data=df, x="Education", y="Investment_risk_willingness",hue= 'Gender', split=True )
plt.title('Correlation between Risk Willingness in Finance and Highest Education with the Gender Comparison from Russians Investors', fontsize=14)
plt.xlabel("Type of Education")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");





df.Education.describe()
df['Education'].value_counts(sort=True)
# CONFIRMED as it is less than 0.05
from scipy.stats import chi2_contingency

contingency1 = [[129, 109],
              [87, 83],
                [53, 109],
                [39, 80],
                [33, 80],
               [16,83]]
chi2_contingency(contingency1)

# df['Investment_risk_willingness'].value_counts(sort=True)

df['Education'].value_counts(sort=True)

# # Emotional_stability AND Investment_risk_willingness



plt.figure(figsize=(10,7))
sns.barplot(x="Emotional_stability", y="Investment_risk_willingness", data=df)
plt.title('Correlation between Risk Willingness in Finance and Emotinal Stability from Russians Investors', fontsize=14)
plt.xlabel("Emotional Stability Levels (1:not at all, 5:extremely much)")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");




plt.figure(figsize=(16, 10))
sns.boxplot(x="Emotional_stability", y="Investment_risk_willingness", data=df)
plt.title('Correlation between Risk Willingness in Finance and Emotinal Stability from Russians Investors', fontsize=14)
plt.xlabel("Emotional Stability Levels (1:not at all, 5:extremely much)")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");





plt.figure(figsize=(16, 10))
sns.violinplot(data=df, x="Emotional_stability", y="Investment_risk_willingness")

plt.title('Correlation between Risk Willingness in Finance and Emotinal Stability from Russians Investors', fontsize=14)
plt.xlabel("Emotional Stability Levels (1:not at all, 5:extremely much)")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");







plt.figure(figsize=(16, 10))
sns.violinplot(data=df, x="Emotional_stability", y="Investment_risk_willingness",hue= 'Gender', split=True )
plt.title('Correlation between Risk Willingness in Finance and Emotinal Stability with the Gender Comparison from Russians Investors', fontsize=14)
plt.xlabel("Emotional Stability Levels (1:not at all, 5:extremely much)")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");






df.Education.describe()
df['Education'].value_counts(sort=True)
# CONFIRMED as it is less than 0.05
from scipy.stats import chi2_contingency

contingency1 = [[141, 109],
              [99, 83],
                [52, 109],
                [38, 83],
                [22, 80],
               ]
chi2_contingency(contingency1)

# df['Emotional_stability'].value_counts(sort=True)

df['Investment_risk_willingness'].value_counts(sort=True)

# # Age AND Investment_risk_willingness


plt.figure(figsize=(10,7))
sns.barplot(x="Age", y="Investment_risk_willingness", data=df)
plt.title('Correlation between Risk Willingness in Finance and Age Groups from Russians Investors', fontsize=14)
plt.xlabel("Age Groups(18-23: 20.5,24-29: 26.5, 30-40:35,41+:41)")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");




plt.figure(figsize=(16, 10))
sns.boxplot(x="Age", y="Investment_risk_willingness", data=df)
plt.title('Correlation between Risk Willingness in Finance and Age Groups from Russians Investors', fontsize=14)
plt.xlabel("Age Groups(18-23: 20.5,24-29: 26.5, 30-40:35,41+:41)")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");







plt.figure(figsize=(16, 10))
sns.violinplot(data=df, x="Age", y="Investment_risk_willingness")
plt.title('Correlation between Risk Willingness in Finance and Age Groups from Russians Investors', fontsize=14)
plt.xlabel("Age Groups(18-23: 20.5,24-29: 26.5, 30-40:35,41+:41)")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");









plt.figure(figsize=(16, 10))
sns.violinplot(data=df, x="Age", y="Investment_risk_willingness",hue= 'Gender', split=True )
plt.title('Correlation between Risk Willingness in Finance and Age Groups with the Gender Comparison from Russians Investors', fontsize=14)
plt.xlabel("Age Groups(18-23: 20.5,24-29: 26.5, 30-40:35,41+:41)")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");

df.Education.describe()
df['Education'].value_counts(sort=True)
# NOT CONFIRMED as it is more than 0.05
from scipy.stats import chi2_contingency

contingency1 = [[127, 109],
              [97, 109],
                [73, 80],
                [63, 83],
               [63,48]]
chi2_contingency(contingency1)

# df['Age'].value_counts(sort=True)

df['Investment_risk_willingness'].value_counts(sort=True)

# # Best_method AND Investment_risk_willingness


plt.figure(figsize=(10,7))
sns.barplot(x="Best_method", y="Investment_risk_willingness", data=df)
plt.title('Correlation between Risk Willingness in Finance and Correlation Methods of Investments from Russians Investors', fontsize=14)
plt.xlabel("Methods of Investing")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");




plt.figure(figsize=(16, 10))
sns.boxplot(x="Best_method", y="Investment_risk_willingness", data=df)
plt.title('Correlation between Risk Willingness in Finance and Correlation Methods of Investments from Russians Investors', fontsize=14)
plt.xlabel("Methods of Investing")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");







plt.figure(figsize=(16, 10))
sns.violinplot(data=df, x="Best_method", y="Investment_risk_willingness")
plt.title('Correlation between Risk Willingness in Finance and Correlation Methods of Investments from Russians Investors', fontsize=14)
plt.xlabel("Methods of Investing")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");









plt.figure(figsize=(16, 10))
sns.violinplot(data=df, x="Best_method", y="Investment_risk_willingness",hue= 'Gender', split=True )
plt.title('Correlation between Risk Willingness in Finance and Correlation Methods of Investments with the Gender Comparison from Russians Investors', fontsize=14)
plt.xlabel("Methods of Investing")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");

df.Education.describe()
# CONFIRMED as it is less than 0.05
from scipy.stats import chi2_contingency

contingency1 = [[119, 109],
              [80, 80],
                [76, 109],
                [35, 83],
                [23, 83],
               [19,83]]
chi2_contingency(contingency1)

# df['Best_method'].value_counts(sort=True)

df['Investment_risk_willingness'].value_counts(sort=True)

# # Analyses AND Investment_risk_willingness


plt.figure(figsize=(16, 10))
sns.barplot(x="Analyses", y="Investment_risk_willingness", data=df)
plt.title('Correlation between Risk Willingness in Finance and Correlation between Familiarity with Technical and Fundamental Analyses from Russians Investors', fontsize=14)
plt.xlabel("Types of Knowing Analyses")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");




plt.figure(figsize=(16, 10))
sns.boxplot(x="Analyses", y="Investment_risk_willingness", data=df)
plt.title('Correlation between Risk Willingness in Finance and Correlation between Familiarity with Technical and Fundamental Analyses from Russians Investors', fontsize=14)
plt.xlabel("Types of Knowing Analyses")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");







plt.figure(figsize=(16, 10))
sns.violinplot(x="Analyses", y="Investment_risk_willingness", data=df)
plt.title('Correlation between Risk Willingness in Finance and Correlation between Familiarity with Technical and Fundamental Analyses from Russians Investors', fontsize=14)
plt.xlabel("Types of Knowing Analyses")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");









plt.figure(figsize=(16, 10))
sns.violinplot(x="Analyses", y="Investment_risk_willingness", data=df, hue= 'Gender', split=True)
plt.title('Correlation between Risk Willingness in Finance and Correlation between Familiarity with Technical and Fundamental Analyses with the Gender Comparison from Russians Investors', fontsize=14)
plt.xlabel("Types of Knowing Analyses")
plt.ylabel("Willingness to Take Risk with Finance (1: Never, 4: Always)");

df.Education.describe()
df['Education'].value_counts(sort=True)
# CONFIRMED as it is less than 0.05
from scipy.stats import chi2_contingency

contingency1 = [[116, 109],
              [107, 109],
                [70, 109],
                [62, 109],
                ]
chi2_contingency(contingency1)

# df['Analyses'].value_counts(sort=True)

df['Investment_risk_willingness'].value_counts(sort=True)

# # Best_method AND Age


plt.figure(figsize=(16, 10))
sns.barplot(x="Best_method", y="Age", data=df)
plt.title('Correlation between Ages and Methods of Investments from Russians Investors', fontsize=14)
plt.xlabel("Methods of Investments")
plt.ylabel("Age")






plt.figure(figsize=(16, 10))
sns.boxplot(x="Best_method", y="Age", data=df)
plt.title('Correlation between Ages and Methods of Investments from Russians Investors', fontsize=14)
plt.xlabel("Methods of Investments")
plt.ylabel("Age")









plt.figure(figsize=(16, 10))
sns.violinplot(x="Best_method", y="Age", data=df)
plt.title('Correlation between Ages and Methods of Investments from Russians Investors', fontsize=14)
plt.xlabel("Methods of Investments")
plt.ylabel("Age")










plt.figure(figsize=(16, 10))
sns.violinplot(x="Best_method", y="Age", data=df, hue= 'Gender', split=True)
plt.title('Correlation between Ages and Methods of Investments with the Gender Comparison from Russians Investors', fontsize=14)
plt.xlabel("Methods of Investments")
plt.ylabel("Age")



df.Education.describe()
df['Education'].value_counts(sort=True)
# CONFIRMED as it is less than 0.05
from scipy.stats import chi2_contingency

contingency1 = [[119, 127],
              [80, 97],
                [76, 97],
                [35, 127],
                [23,127],[19,97]]
chi2_contingency(contingency1)

# df['Best_method'].value_counts(sort=True)

df['Age'].value_counts(sort=True)

# # Best_method AND Age


plt.figure(figsize=(16, 10))
sns.barplot(x="Education", y="Age", data=df)
plt.title('Correlation between Ages and Highest Education from Russians Investors', fontsize=14)
plt.xlabel("Type of Education")
plt.ylabel("Age")






plt.figure(figsize=(16, 10))
sns.boxplot(x="Education", y="Age", data=df)
plt.title('Correlation between Ages and Highest Education from Russians Investors', fontsize=14)
plt.xlabel("Type of Education")
plt.ylabel("Age")









plt.figure(figsize=(16, 10))
sns.violinplot(x="Education", y="Age", data=df)
plt.title('Correlation between Ages and Highest Education from Russians Investors', fontsize=14)
plt.xlabel("Type of Education")
plt.ylabel("Age")










plt.figure(figsize=(16, 10))
sns.violinplot(x="Education", y="Age", data=df, hue= 'Gender', split=True)
plt.title('Correlation between Ages and Highest Education with the Gender Comparison from Russians Investors', fontsize=14)
plt.xlabel("Methods of Investments")
plt.ylabel("Age")




df.Education.describe()
df['Education'].value_counts(sort=True)
# CONFIRMED as it is less than 0.05
from scipy.stats import chi2_contingency

contingency1 = [[129, 127],
              [87, 127],
                [53, 73],
                [33, 97],
                [16,63]]
chi2_contingency(contingency1)

# df['Education'].value_counts(sort=True)

df['Age'].value_counts(sort=True)

df.head()





















































































