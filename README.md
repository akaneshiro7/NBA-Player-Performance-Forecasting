# NBA Player Performance Forecasting

Dataset from https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats/data?select=Player+Per+Game.csv

## Abstract
Our team’s project will be conducted with the intention of forecasting NBA player’s performance based on statistics and data from their previous seasons. Our team would apply time-series modeling, clustering, Kalman-filters and other machine learning algorithms to our dataset in order to predict player performance. For sports managers and team personnel, this project could be applied to provide additional insights regarding trades and team construction. For sports enthusiasts and sports bettors, this project could be used to make optimal, data-driven decisions, aid in fantasy-sports team constructions or uncover theoretical arbitrage opportunities. 

To conduct this project, we have sourced data from professional men’s basketball players’ stats from 1997-2023. This publicly-available dataset includes 26 years of players’ season-averages including points, rebounds, and assists, as well as in-depth tracking data like shot volume by location and positional playing time. These features provide a clear view of each player’s performance, season over season. Counting stats are the most basic measures of a player’s in-game actions: points, rebounds, and assists totals are frequently used as go-to metrics to quickly assess a player’s scoring or passing skill. Tracking data provides more insight into each player’s role on the court and how they reach the counting totals they do. This mix of high- and low-level features, alongside age and experience data, will provide a year-by-year look at each player’s production, which can then be sequenced to give a cohesive look at how a player’s statistical profile changes over time. 

The project will look to predict season-wide player performance in the three most important counting-stat categories: points, rebounds, and assists. Using knowledge of their previous seasons’ performance and where they are in their careers (based on their age and number of seasons played) an end-of-season statline will be generated of all players with at least three years of experience. Firstly, principal component analysis (PCA), will be employed to reduce the dataset’s high dimensionality. The statistical predictions will be done using different time-series models, like Kalman filtering and feed-forward neural networks. Classification models like Bayes’ decision rule and logistic regression will look to answer categorical questions about how a player will perform relative to previous seasons, like predicting whether or not they will match their previous season’s point totals or if they will significantly increase their number of assists from two years ago. 

Our team is seeking to understand how players progress and regress over the course of their careers. With our models, we will try to quantitatively predict end-of-season statistical totals, but also which young players are candidates for breakout seasons and who’s on the cusp of an age-based downswing. As such, we hope to grow our understanding of players’ aging curves and the relationships between their roles on the court with the stability of their statistical output. The National Basketball Association is a $10B industry that has grown increasingly analytically-driven over the last decade. Advancements in motion tracking technology have given teams more data than ever before to make tactical and personnel decisions. Every organization is looking for a competitive edge in building their roster; our project will look to provide insight on how to best do so. This project can also be applied in sports betting contexts. Game-by-game and season-long counting stat totals are a very common betting category, which makes an accurate model extremely valuable to bettors and houses alike in a multi-billion dollar, rapidly growing industry. As sports grow increasingly data-driven, predictive models like ours will grow increasingly important for key decision-makers in and around the game. 
