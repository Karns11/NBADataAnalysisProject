from bs4 import BeautifulSoup
import requests

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pandas as pd

#-----------------------------------------------------------------------------#
#THIS SECTION WILL GRAB THE RAW SCORES AND DATESM IN THE FORM OF: [DATE, TEAM1, TEAM1 PTS, TEAM2, TEAM2 PTS]

year = 2023
#empty list that we are going to add each months scores into
total_stats = []
dates = []

oct_url = "https://www.basketball-reference.com/leagues/NBA_2023_games-october.html".format(year)


response = requests.get(oct_url)
html_oct = response.text
soup = BeautifulSoup(html_oct, 'html.parser')

the_tr_tags = soup.find_all('tr')
the_headers = [th.getText() for th in the_tr_tags[0].find_all('th')]

the_headers = the_headers[1:]
the_headers[1] = 'Visitor/N'
the_headers[2] = 'Visitor PTS'
the_headers[3] = 'Home/N'
the_headers[4] = 'Home PTS'

date = [[a.getText() for a in the_tr_tags[i].find_all('a')] for i in range(len(the_tr_tags))]
for item in date:
    if len(item) > 0:
        dates.append(item[0])


data_rows = soup.find_all('tr')[1:]
oct_results = [[td.getText() for td in data_rows[i].find_all('td')] for i in range(len(data_rows))]

#add oct scores to total results
for item in oct_results:
    total_stats.append(item)


nov_url = "https://www.basketball-reference.com/leagues/NBA_{}_games-november.html".format(year)

response = requests.get(nov_url)
html_nov = response.text
soup = BeautifulSoup(html_nov, 'html.parser')

the_tr_tags = soup.find_all('tr')
date = [[a.getText() for a in the_tr_tags[i].find_all('a')] for i in range(len(the_tr_tags))]
for item in date:
    if len(item) > 0:
        dates.append(item[0])


data_rows = soup.find_all('tr')[1:]
nov_results = [[td.getText() for td in data_rows[i].find_all('td')] for i in range(len(data_rows))]
for item in nov_results:
    total_stats.append(item)
    

dec_url = "https://www.basketball-reference.com/leagues/NBA_{}_games-december.html".format(year)

response = requests.get(dec_url)
html_dec = response.text
soup = BeautifulSoup(html_dec, 'html.parser')

the_tr_tags = soup.find_all('tr')
date = [[a.getText() for a in the_tr_tags[i].find_all('a')] for i in range(len(the_tr_tags))]
for item in date:
    if len(item) > 0 and len(item) > 3:
        dates.append(item[0])

data_rows = soup.find_all('tr')[1:]
dec_results = [[td.getText() for td in data_rows[i].find_all('td')] for i in range(len(data_rows))]
for item in dec_results:
    if item[2] != '':
        total_stats.append(item)   

#print(total_stats)

#raw_scores_df = [Date, Team1, Team1 PTS, team2, Team2 PTS]
schedule_results_df = pd.DataFrame(total_stats, columns=the_headers)
schedule_results_df.insert(0, 'Date', dates)
necessary_data_df = schedule_results_df.loc[:, ['Date', 'Visitor/N', 'Visitor PTS', 'Home/N', 'Home PTS']]
raw_scores_df = necessary_data_df.rename(columns={'Visitor/N': "Team1", "Visitor PTS": "Team1 PTS", "Home/N": "Team2", "Home PTS": "Team2 PTS"})
#print(raw_scores_df)
#print()

#raw_scores_df.to_excel('2023GameResults.xlsx', sheet_name='Sheet1')

#-----------------------------------------------------------------------------#
#THIS SECTION GETS THE TEAM STATS PER GAME

year = 2023

the_url = "https://www.basketball-reference.com/leagues/NBA_{}.html".format(year)

response = requests.get(the_url)
html_the_team = response.text
soup = BeautifulSoup(html_the_team, 'html.parser')

the_tr = soup.find_all('tr')

per_game_stats = [th.getText() for th in the_tr[72].find_all('th')]
per_game_stats = per_game_stats[1:]

total_stats = [th.getText() for th in the_tr[200].find_all('th')]
total_stats = total_stats[1:]


per100_stats = [th.getText() for th in the_tr[200].find_all('th')]
per100_stats = per100_stats[1:]




the_team_rows = soup.find_all('tr')
the_data_raw = [[td.getText()for td in the_team_rows[i].find_all('td')] for i in range(len(the_team_rows))]
the_data = the_data_raw[73:]
#print(the_data)
clean_data = []

for row in the_data:
    if len(row) > 0:
        clean_row = []
        team = row[0].rstrip('*')
        clean_row.append(team)
        for val in row[1:]:
            if val != '':
                clean_row.append(val)
            else:
                clean_row.append(None)
        clean_data.append(clean_row)
        
#print(clean_data)
#for row in clean_data:
    #print(row[0])

#this has the team stats per game in a list
per_game_team_stats = clean_data[:30]
#print(per_game_team_stats)
per_game_team_df = pd.DataFrame(per_game_team_stats, columns=per_game_stats)
per_game_team_df.name = '2023 Team Per Game Stats'
#print(per_game_team_df.name)
#print(per_game_team_df)
#print()
#per_game_team_df.to_excel('2023TeamPerGameStats.xlsx', sheet_name='Sheet1')

#-----------------------------------------------------------------------------#
#THIS SECTION GETS A TEAMS OPPENENT STATS PER GAME
#this has the team opponent stats per game in a list
per_game_opp_stats = the_data[32:62]
clean_per_game_opp_stats = []

for row in per_game_opp_stats:
    if len(row) > 0:
        new_clean_row = []
        team = row[0].rstrip('*')
        new_clean_row.append(team)
        for val in row[1:]:
            if val != '':
                new_clean_row.append(val)
            else:
                new_clean_row.append(None)
        clean_per_game_opp_stats.append(new_clean_row)

#print(clean_per_game_opp_stats)

per_game_opp_df = pd.DataFrame(clean_per_game_opp_stats, columns=per_game_stats)
per_game_opp_df.name = '2023 Opp Per Game Stats'
#print(per_game_opp_df.name)
#print(per_game_opp_df)
#print()
#per_game_opp_df.to_excel('2023OppPerGameStats.xlsx', sheet_name='Sheet1')


#-----------------------------------------------------------------------------#
combine = pd.merge(per_game_team_df, per_game_opp_df, on='Team', how='outer')
combined_df = combine

team1_df = combined_df.rename(columns={"Team": "Team1", "G_x": "Team1GP", "MP_x": "Team1MP", "FG_x": "Team1FG", "FGA_x": "Team1FGA", "FG%_x": "Team1FG%", "3P_x": "Team13P", "3PA_x": "Team13PA", "3P%_x": "Team13P%", "2P_x": "Team12P", "2PA_x": "Team12PA", "2P%_x": "Team12P%", "FT_x": "Team1FT", "FTA_x": "Team1FTA", "FT%_x": "Team1FT%", "ORB_x": "Team1ORB", "DRB_x": "Team1DRB", "TRB_x": "Team1TRB", "AST_x": "Team1AST", "STL_x": "Team1STL", "BLK_x": "Team1BLK", "TOV_x": "Team1TOV", "PF_x": "Team1PF", "PTS_x": "Team1PTS", "G_y": "OppTeam1GP", "MP_y": "OppTeam1MP", "FG_y": "OppTeam1FG", "FGA_y": "OppTeam1FGA", "FG%_y": "OppTeam1FG%", "3P_y": "OppTeam13P", "3PA_y": "OppTeam13PA", "3P%_y": "OppTeam13P%", "2P_y": "OppTeam12P", "2PA_y": "OppTeam12PA", "2P%_y": "OppTeam12P%", "FT_y": "OppTeam1FT", "FTA_y": "OppTeam1FTA", "FT%_y": "OppTeam1FT%", "ORB_y": "Team1ORB", "DRB_y": "OppTeam1DRB", "TRB_y": "OppTeam1TRB", "AST_y": "OppTeam1AST", "STL_y": "OppTeam1STL", "BLK_y": "OppTeam1BLK", "TOV_y": "OppTeam1TOV", "PF_y": "OppTeam1PF", "PTS_y": "OppTeam1PTS"})
#print(team1_df)
#team1_df.to_excel('Team1_2023CombinedTeamAndOppStats.xlsx', sheet_name='Sheet1')

team2_df = combined_df.rename(columns={"Team": "Team2", "G_x": "Team2GP", "MP_x": "Team2MP", "FG_x": "Team2FG", "FGA_x": "Team2FGA", "FG%_x": "Team2FG%", "3P_x": "Team23P", "3PA_x": "Team23PA", "3P%_x": "Team23P%", "2P_x": "Team22P", "2PA_x": "Team22PA", "2P%_x": "Team22P%", "FT_x": "Team2FT", "FTA_x": "Team2FTA", "FT%_x": "Team2FT%", "ORB_x": "Team2ORB", "DRB_x": "Team2DRB", "TRB_x": "Team2TRB", "AST_x": "Team2AST", "STL_x": "Team2STL", "BLK_x": "Team2BLK", "TOV_x": "Team2TOV", "PF_x": "Team2PF", "PTS_x": "Team2PTS", "G_y": "OppTeam2GP", "MP_y": "OppTeam2MP", "FG_y": "OppTeam2FG", "FGA_y": "OppTeam2FGA", "FG%_y": "OppTeam2FG%", "3P_y": "OppTeam23P", "3PA_y": "OppTeam23PA", "3P%_y": "OppTeam23P%", "2P_y": "OppTeam22P", "2PA_y": "OppTeam22PA", "2P%_y": "OppTeam22P%", "FT_y": "OppTeam2FT", "FTA_y": "OppTeam2FTA", "FT%_y": "OppTeam2FT%", "ORB_y": "Team2ORB", "DRB_y": "OppTeam2DRB", "TRB_y": "OppTeam2TRB", "AST_y": "OppTeam2AST", "STL_y": "OppTeam2STL", "BLK_y": "OppTeam2BLK", "TOV_y": "OppTeam2TOV", "PF_y": "OppTeam2PF", "PTS_y": "OppTeam2PTS"})
#print(team2_df)
#team2_df.toexcel('Team2_2023CombinedTeamAndOppStats.xlsx', sheet_name='Sheet1')

#-----------------------------------------------------------------------------#
combineT1 = pd.merge(raw_scores_df, team1_df, on="Team1", how="outer")
#print(combineT1)
#combineT1.to_excel("ResultsWithTeam1Stats.xlsx", sheet_name='Sheet1')

#-----------------------------------------------------------------------------#
#NEW_DF IS THE MAIN DATA FRAME THAT I AM WORKING WITH

results_with_stats_df = pd.merge(combineT1, team2_df, on="Team2", how="outer")

#results_with_stats_df.to_excel("teamstatstest.xlsx", sheet_name='Sheet1')
#results_with_stats_df[['Team1 PTS', 'Team2 PTS', 'Team1GP', 'Team1MP',	'Team1FG',	'Team1FGA',	'Team1FG%',	'Team13P',	'Team13PA',	'Team13P%',	'Team12P',	'Team12PA',	'Team12P%',	'Team1FT',	'Team1FTA',	'Team1FT%', 'Team1ORB',	'Team1DRB',	'Team1TRB',	'Team1AST',	'Team1STL',	'Team1BLK',	'Team1TOV',	'Team1PF',	'Team1PTS',	'OppTeam1GP',	'OppTeam1MP',	'OppTeam1FG',	'OppTeam1FGA',	'OppTeam1FG%',	'OppTeam13P',	'OppTeam13PA',	'OppTeam13P%',	'OppTeam12P',	'OppTeam12PA',	'OppTeam12P%',	'OppTeam1FT',	'OppTeam1FTA',	'OppTeam1FT%',	'Team1ORB',	'OppTeam1DRB',	'OppTeam1TRB',	'OppTeam1AST',	'OppTeam1STL',	'OppTeam1BLK',	'OppTeam1TOV',	'OppTeam1PF',	'OppTeam1PTS',	'Team2GP',	'Team2MP',	'Team2FG',	'Team2FGA',	'Team2FG%',	'Team23P',	'Team23PA',	'Team23P%',	'Team22P',	'Team22PA',	'Team22P%',	'Team2FT',	'Team2FTA',	'Team2FT%',	'Team2ORB',	'Team2DRB',	'Team2TRB',	'Team2AST',	'Team2STL',	'Team2BLK',	'Team2TOV',	'Team2PF',	'Team2PTS',	'OppTeam2GP',	'OppTeam2MP',	'OppTeam2FG',	'OppTeam2FGA',	'OppTeam2FG%',	'OppTeam23P',	'OppTeam23PA',	'OppTeam23P%',	'OppTeam22P',	'OppTeam22PA',	'OppTeam22P%',	'OppTeam2FT',	'OppTeam2FTA',	'OppTeam2FT%',	'Team2ORB',	'OppTeam2DRB',	'OppTeam2TRB',	'OppTeam2AST',	'OppTeam2STL',	'OppTeam2BLK',	'OppTeam2TOV',	'OppTeam2PF',	'OppTeam2PTS']] = results_with_stats_df[['Team1 PTS', 'Team2 PTS', 'Team1GP', 'Team1MP',	'Team1FG',	'Team1FGA',	'Team1FG%',	'Team13P',	'Team13PA',	'Team13P%',	'Team12P',	'Team12PA',	'Team12P%',	'Team1FT',	'Team1FTA',	'Team1FT%', 'Team1ORB',	'Team1DRB',	'Team1TRB',	'Team1AST',	'Team1STL',	'Team1BLK',	'Team1TOV',	'Team1PF',	'Team1PTS',	'OppTeam1GP',	'OppTeam1MP',	'OppTeam1FG',	'OppTeam1FGA',	'OppTeam1FG%',	'OppTeam13P',	'OppTeam13PA',	'OppTeam13P%',	'OppTeam12P',	'OppTeam12PA',	'OppTeam12P%',	'OppTeam1FT',	'OppTeam1FTA',	'OppTeam1FT%',	'Team1ORB',	'OppTeam1DRB',	'OppTeam1TRB',	'OppTeam1AST',	'OppTeam1STL',	'OppTeam1BLK',	'OppTeam1TOV',	'OppTeam1PF',	'OppTeam1PTS',	'Team2GP',	'Team2MP',	'Team2FG',	'Team2FGA',	'Team2FG%',	'Team23P',	'Team23PA',	'Team23P%',	'Team22P',	'Team22PA',	'Team22P%',	'Team2FT',	'Team2FTA',	'Team2FT%',	'Team2ORB',	'Team2DRB',	'Team2TRB',	'Team2AST',	'Team2STL',	'Team2BLK',	'Team2TOV',	'Team2PF',	'Team2PTS',	'OppTeam2GP',	'OppTeam2MP',	'OppTeam2FG',	'OppTeam2FGA',	'OppTeam2FG%',	'OppTeam23P',	'OppTeam23PA',	'OppTeam23P%',	'OppTeam22P',	'OppTeam22PA',	'OppTeam22P%',	'OppTeam2FT',	'OppTeam2FTA',	'OppTeam2FT%',	'Team2ORB',	'OppTeam2DRB',	'OppTeam2TRB',	'OppTeam2AST',	'OppTeam2STL',	'OppTeam2BLK',	'OppTeam2TOV',	'OppTeam2PF',	'OppTeam2PTS']].apply(pd.to_numeric)
new_df = results_with_stats_df.drop(columns=['Date', 'Team1', 'Team2'], axis=1)
new_df.insert(2, 'Team1 Win', 0)
new_df = new_df.astype(float)
team1_win = new_df['Team1 PTS'].gt(new_df['Team2 PTS'])
new_df['Team1 Win'] = team1_win
new_df['Team1 Win'] = new_df['Team1 Win'].astype(int)
new_df = new_df.drop(columns=['Team1 PTS', 'Team2 PTS', 'Team1GP' ,'Team1MP', 'OppTeam1GP', 'OppTeam1MP', 'Team2GP', 'Team2MP', 'Team2MP', 'OppTeam2MP', 'OppTeam2GP'], axis=1)
#print(new_df)
#print()
#new_df.to_excel("ResultsWithTeam1And2Stats.xlsx", sheet_name='Sheet1')
#-----------------------------------------------------------------------------#
#THIS IS THE DATA USED FOR PREDICTIONS

def as_list(row):
    return list(row)

all_data = [ row for row in team1_df.apply(as_list, axis=1)]

to_remove =[1, 2, 24, 25]
pred_data = []

for team in all_data:
    for index in to_remove:
        del team[index]
        

for item in all_data:
    pred_data.append(item)
    
def get_first(item):
    return item[0]

sorted_pred_data = sorted(pred_data, key=get_first)      
#print(sorted_pred_data)


use_this = []
for ind, team in enumerate(sorted_pred_data):
    team = team[1:]
    use_this.append(sorted_pred_data[29])
    use_this.append(sorted_pred_data[13])
    break
#print(use_this)

#-----------------------------------------------------------------------------#
from sklearn.linear_model import LinearRegression


X = new_df.drop('Team1 Win', axis=1)
#print(X)
#X.to_excel("X.xlsx", sheet_name='Sheet1')
y = new_df['Team1 Win']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the test data
score = model.score(X_test, y_test)

###print(f'LinearRegression Test score: {score:.2f}')



import numpy as np
from sklearn.svm import SVC
model1 = SVC()

# Train the model on the training data
model1.fit(X_train, y_train)

# Evaluate the model on the test data
score = model1.score(X_test, y_test)

print()
print(f'SVC Test score: {score:.2f}')
print()

new_data = use_this[0]+use_this[1]
new_data.pop(0)
new_data.pop(42)
#print(new_data)
array = np.array(new_data)
thing = array.astype(np.float64)
test_data = []
test_data.append(thing)


predictions = model.predict(X_test)

#print(predictions)
#print()

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")
print()

comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
#print(comparison_df)
#print()
#comparison_df.to_excel('comparison_results.xlsx', index=False)

# Function to determine correctness based on conditions
def mark_correctness(row):
    if row['Actual'] == 0 and row['Predicted'] < 0.5:
        return 'Correct'
    elif row['Actual'] == 1 and row['Predicted'] > 0.5:
        return 'Correct'
    else:
        return 'Wrong'

# Apply the function to create a new column 'Correctness'
comparison_df['Correctness'] = comparison_df.apply(mark_correctness, axis=1)

#print(comparison_df)
#print()

correct_counts = comparison_df['Correctness'].value_counts()
total_rows = len(comparison_df)
#print(correct_counts)
#print()

percentage_correct = correct_counts['Correct'] / total_rows * 100
print(f"Overall Percentage of Correct Predictions: {percentage_correct:.2f}%")




from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the test data
score = model.score(X_test, y_test)

###print(f'Decision Tree Test score: {score:.2f}')





from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the test data
score = model.score(X_test, y_test)

###print(f'RandomForest Test score: {score:.2f}')





from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the test data
score = model.score(X_test, y_test)

###print(f'KNN Test score: {score:.2f}')
