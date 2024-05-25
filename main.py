from selenium.webdriver import Edge
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
import selenium.webdriver.support.expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
import re
import pandas as pd

start_uri = "https://en.volleyballworld.com/volleyball/competitions/volleyball-nations-league/2021/schedule/#fromDate=2021-05-25"
match_id = 1
options = Options()
service = Service('msedgedriver.exe')
browser = Edge(options=options, service=service)

browser.get(start_uri)
browser.implicitly_wait(5)

# click to close cookies request
browser.execute_script(
    'arguments[0].click();',
    browser.find_element('xpath', '//*[@id="onetrust-close-btn-container"]/button')
)

# click to open gender dropdown
browser.execute_script(
    'arguments[0].click();',
    browser.find_element(By.XPATH, '/html/body/div[1]/main/section/div[1]/div/div[3]/div[2]/div[1]/div/div[3]/div')
)

# select 'woman' in dropdown
browser.execute_script(
    'arguments[0].click();',
    browser.find_element('xpath', '/html/body/div[1]/main/section/div[1]/div/div[3]/div[2]/div[1]/div/div[3]/ul/li[2]')
)

# get all cards of matches
matches_card = browser.find_element(
    By.CSS_SELECTOR, 
    'div.vbw-gs2-match-data-card'
)

match_url = matches_card.find_element('css selector', 'a').get_attribute('href')
browser.get(match_url)
browser.implicitly_wait(4)


def set_row(columns, values):    
    for col, value in zip(columns, values):
        lst_idx = stats_df_clean.last_valid_index()
        lst_idx = lst_idx+1 if lst_idx is not None else 0
        stats_df_clean.loc[match_id, col] = value


stats_dataframes = pd.DataFrame()
while True:
    pool, phase, matchN = browser.find_element('xpath', '//*[@id="main-content"]/section/div/div/div[2]/div/a[1]/div[2]/div[1]').text.split(' - ')
    pool = re.sub('\D+', '', pool)
    matchN = re.sub('\D+', '', matchN)

    arena = re.sub(
        r'\n', 
        '-', 
        browser.find_element('xpath', '//*[@id="main-content"]/section/div/div/div[2]/div/a[1]/div[2]/div[3]').text
    )

    home_team = browser.find_element('xpath', '//*[@id="main-content"]/section/div/div/div[2]/div/a[2]/div[1]/div[1]/div[2]').get_attribute('innerHTML')
    away_team = browser.find_element('xpath', '//*[@id="main-content"]/section/div/div/div[2]/div/a[2]/div[1]/div[3]/div[2]').get_attribute('innerHTML')

    home_res, away_res = browser.find_element('xpath', '//*[@id="main-content"]/section/div/div/div[2]/div/a[2]/div[1]/div[2]/div[1]').text.split('\n:\n')

    table = browser.find_element('xpath', '/html/body/div[1]/main/div/section/div/div[1]/div[1]/div[2]/div/table')
    stats_table = pd.read_html(table.get_attribute('outerHTML'))[0]

    stats_colsA = (stats_table['Match StatsStats'] + ' A').tolist()
    stats_colsB = (stats_table['Match StatsStats'] + ' B').tolist()
    stats_colsA[-1] = stats_colsA[-1]+'2'
    stats_colsB[-1] =  stats_colsB[-1]+'2'

    stats_df_clean = pd.DataFrame(columns=[*stats_colsA, *stats_colsB])

    stats_df_clean.loc[match_id, 'match_id'] = match_id
    stats_df_clean.loc[match_id, 'home'] = home_team
    stats_df_clean.loc[match_id, 'away'] = away_team
    stats_df_clean.loc[match_id, 'home_res'] = home_res
    stats_df_clean.loc[match_id, 'away_res'] = away_res

    stats_df_clean.loc[match_id, 'pool'] = pool
    stats_df_clean.loc[match_id, 'phase'] = phase
    stats_df_clean.loc[match_id, 'matchN'] = matchN
    stats_df_clean.loc[match_id, 'arena'] = arena
    set_row(stats_colsA, stats_table.iloc[:,1])
    set_row(stats_colsB, stats_table.iloc[:,3])

    stats_dataframes = pd.concat([stats_dataframes, stats_df_clean], axis=0)

    upcomming = browser.find_element('xpath', '//*[@id="main-content"]/div/section/div/div[2]/section/div[2]/div[1]/div/div[1]/a').get_attribute('innerHTML')
    if upcomming != 'Match Centre':
        break

    match_id += 1
    browser.execute_script(
        'arguments[0].click();',
        browser.find_element('xpath', '//*[@id="main-content"]/div/section/div/div[2]/section/div[2]/div[1]/div/div[1]/a')
    )
    browser.implicitly_wait(3)

stats_dataframes.to_excel('match_stats.xlsx')
browser.quit()
