# Electricity classification

## Background
The Sustainable Development Goals (SDGs), also known as the Global Goals, were adopted by the United Nations in 2015 as a universal call to action to end poverty, protect the planet, and ensure that by 2030 all people enjoy peace and prosperity.

![Sustainable Development Goals](https://i1.wp.com/www.un.org/sustainabledevelopment/wp-content/uploads/2015/12/english_SDG_17goals_poster_all_languages_with_UN_emblem_1.png?fit=728%2C451&ssl=1)

Data for each of these goas is available from teh UN Statistics database. The aim on this project is to predicting dominant energy source by country using the SDG indicator dat.

## About the data

The data used in this is from the United Nations' database. 
- Features consist of indicators for the UN's Sustainable Development Goals initiative by geographical in the year 2019. 
- Targets are the dominant energy source per geographical area.
- The data for the start of this project consists of 85 geographical areas for which target data was available, and 3 indicators per area. Indcators are the percentage population with access to electricity in:
   - rural areas.
   - urban areas.
   - all areas.

This is a multiclass classification problem. Each target is one of 5 classes:

| Code | Label |
|------|-------|
|  0   |  combustible |
|  1   |  hydro |
|  2   |  other |
|  3   |  solar |
|  4   |  wind |

## Results
Models evaluated thus far including the score achieved on the validation and training sets are displayed in the table below.

![data](/presentation/images/results.png)