<h1 style="text-align:center;"> Volleyball Nations League Scraper </h1>

<h2 style="margin-left: 5%;">
    A Python script to scrape VNL data from the <a href="https://en.volleyballworld.com/">official website</a>
</h2>

## How it works
   The script was written using Python, pandas, and selenium. All the execution of the program is in the file `main.py`. Simply instantiate the `Scraper` class and provide the necessary arguments.
   
   To scrape all data from the year 2021, use:
   ```python
   scraper = Scraper(
       year=2021, 
       from_date='2021-05-25', 
       initial_match_id=0, # optional. Default is 1
       gender='men'  # optional. Default is 'women'
   )

   scraper.run()
   ```

   The data will be saved as .xlsx files with the pattern ```'{self.gender}_match_stats_{year}_{datetime.now().date()}.xlsx'``` for matches and ```'{self.gender}_players_stats_{year}_{datetime.now().date()}.xlsx'``` for players.

   You also can to scrape multiple years using:
   ```python
   Scraper.multiple_years(
        years=[2021, 2022],
        dates=['2021-05-25', '2022-05-30']
   )
   ```

## Notes
   It will be necessary to clean the dataframes. Some data, such as team or player names, may be null because some teams have more players than others, and the tables on the website have a lot of missing values.
