# predict-results-of-football-matches
This project collects data from previous Premier League matches, processes them appropriately and uses machine learning techniques to predict outcomes from future matches.
The code was written in Python


## Dataset
In the ipynb web scraper file there is the code that was used to download the data for Premier league matches from the year 2015-2016 to 2021-2022, this data was received from the https://fbref.com/en/comps/9/Premier-League-Stats website. That's where the name, goals, possession, total passes, total shots on target, fouls and crossovers of each team were taken.
This data was the stats for each match, so it makes no sense to use this data to predict the outcome of a game that has already been made.

In the ipynb PreProccessing file, a new dataset was created from the statistics taken from the fbref website. Initially, the average number of goals, possession, passes, shots on target, fouls and crosses was found for the home and away team until the match they will play, based on the previous matches. Then was calculated the points that each team has in all matches, the points that the home team has in all previous home matches and the away team in all away matches, the points from the last 5 home matches for the home team and the points from the last 5 away games for the away team. The average number of goals scored in home matches for the home team and the average goals in away matches for the away team were also calculated. Finally, a column was added for the values of each group, as given by the website https://www.transfermarkt.com/vereins-statistik/wertvollstemannschaften/marktwertetop/plus/0/galerie/0?land_id=189&kontinent_id=0&yt0=Show.



## Model Selection
Because in the first 5 fixtures there are not enough statistics for the teams, these fictures were deleted for every football year. In the next step with the help of a correlation matrix were found the features with the biggest correlation, and theone of the two was deleted. Created a function that tuning the hyper-parameters of the algorithms and at the end was selected the algorithm with the best accuracy.
