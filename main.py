from selenium.webdriver import Edge
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
import selenium.webdriver.support.expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait, T
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException
from datetime import datetime
from time import sleep
from typing import Literal, Any, Callable

import re
import pandas as pd
import numpy as np
import logging


class Scraper:
    VALID_GENDERS = ('men', 'women')
    def __init__(self, year: int, from_date: str, initial_mid: int=1, gender: Literal['men', 'woman']='women', headless=True) -> None:
        """
        Args:
            year (int): year of the competition.
            from_date (str): initial date.
            initial_mid (int, optional): initial match id. Defaults to 1.
            gender (Literal['men', 'woman'], optional): the gender to scrap. Defaults to 'women'.
            headless (bool, optional): if true launch browser in headless mode
        
        Ex:
        >>> scraper = Scraper(2023, '2023-05-30', gender='men')
        """
        if gender not in self.VALID_GENDERS:
            raise ValueError(f'gender argument must be in {self.VALID_GENDERS}')
        
        self.start_uri = f"https://en.volleyballworld.com/volleyball/competitions/volleyball-nations-league/{year}/schedule/#fromDate={from_date}&gender={gender}"
        self.match_id = initial_mid
        self.gender = gender

        self.matches_dataframe = pd.DataFrame()
        self.stats_dataframe_name = f'{self.gender}_match_stats_{year}_{datetime.now().date()}.xlsx'
        self.match_cols = []

        self.players_dataframe = pd.DataFrame()
        self.players_dataframe_name = f'{self.gender}_players_stats_{year}_{datetime.now().date()}.xlsx'
        self.players_cols = [
            'players_H',
            'Player No_H',
            'Position_H',
            'Total ABS_SCORING_H',
            'Attack Points_SCORING_H',
            'Block Points_SCORING_H',
            'Serve Points_SCORING_H',
            'Errors_SCORING_H',
            'Efficiency %_SCORING_H',
            'Player No_ATTACK_H',
            'Position_ATTACK_H',
            'Point_ATTACK_H',
            'Errors_ATTACK_H',
            'Attempts_ATTACK_H',
            'Total_ATTACK_H',
            'Efficiency %_ATTACK_H',
            'Player No_BLOCK_H',
            'Position_BLOCK_H',
            'Point_BLOCK_H',
            'Errors_BLOCK_H',
            'Touches_BLOCK_H',
            'Total_BLOCK_H',
            'Efficiency %_BLOCK_H',
            'Player No_SERVE_H',
            'Position_SERVE_H',
            'Point_SERVE_H',
            'Errors_SERVE_H',
            'Attempts_SERVE_H',
            'Total_SERVE_H',
            'Efficiency %_SERVE_H',
            'Player No_RECEPTION_H',
            'Position_RECEPTION_H',
            'Successful_RECEPTION_H',
            'Errors_RECEPTION_H',
            'Attempts_RECEPTION_H',
            'Total_RECEPTION_H',
            'Efficiency %_RECEPTION_H',
            'Player No_DIG_H',
            'Position_DIG_H',
            'DigsDig_DIG_H',
            'Errors_DIG_H',
            'Total_DIG_H',
            'Efficiency %_DIG_H',
            'Player No_SET_H',
            'Position_SET_H',
            'Point_SET_H',
            'Errors_SET_H',
            'Attempts_SET_H',
            'Total_SET_H',
            'Efficiency %_SET_H',
            'team_H',
            
            'players_A',
            'Player No_A',
            'Position_A',
            'Total ABS_SCORING_A',
            'Attack Points_SCORING_A',
            'Block Points_SCORING_A',
            'Serve Points_SCORING_A',
            'Errors_SCORING_A',
            'Efficiency %_SCORING_A',
            'Player No_ATTACK_A',
            'Position_ATTACK_A',
            'Point_ATTACK_A',
            'Errors_ATTACK_A',
            'Attempts_ATTACK_A',
            'Total_ATTACK_A',
            'Efficiency %_ATTACK_A',
            'Player No_BLOCK_A',
            'Position_BLOCK_A',
            'Point_BLOCK_A',
            'Errors_BLOCK_A',
            'Touches_BLOCK_A',
            'Total_BLOCK_A',
            'Efficiency %_BLOCK_A',
            'Player No_SERVE_A',
            'Position_SERVE_A',
            'Point_SERVE_A',
            'Errors_SERVE_A',
            'Attempts_SERVE_A',
            'Total_SERVE_A',
            'Efficiency %_SERVE_A',
            'Player No_RECEPTION_A',
            'Position_RECEPTION_A',
            'Successful_RECEPTION_A',
            'Errors_RECEPTION_A',
            'Attempts_RECEPTION_A',
            'Total_RECEPTION_A',
            'Efficiency %_RECEPTION_A',
            'Player No_DIG_A',
            'Position_DIG_A',
            'DigsDig_DIG_A',
            'Errors_DIG_A',
            'Total_DIG_A',
            'Efficiency %_DIG_A',
            'Player No_SET_A',
            'Position_SET_A',
            'Point_SET_A',
            'Errors_SET_A',
            'Attempts_SET_A',
            'Total_SET_A',
            'Efficiency %_SET_A',
            'team_A',
            'match_id',
        ]
        
        options = Options()
        if headless:
            options.add_argument('--headless')
        
        service = Service('msedgedriver.exe')
        
        self.browser = Edge(options=options, service=service)

        self.all_match_links = []

        logging.getLogger('selenium').setLevel(logging.CRITICAL)
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s :: %(levelname)s :: %(module)s.%(funcName)s :: %(message)s",
            encoding='utf-8',
            level=logging.DEBUG,
            filename='logs.log',
            filemode='a'
        )

    def run(self) -> None:
        """Run the program calling all the functions

        Ex:
        >>> scraper = Scraper(2023, '2023-05-30')
        >>> scraper.run()
        """
        self.browser.get(self.start_uri)
        self.browser.maximize_window()

        self.close_cookie_request()
        
        self.get_match_links()
        self.parse_matches()

        self.rename_columns()
        self.save_dfs()

    def quit_browser(self) -> None:
        """Closes the browser and shuts down the WebDriver executable.
        """
        self.browser.quit()
        self.logger.info('Browser quitted')
    
    def set_row(self, df: pd.DataFrame, columns: np.ndarray, values: Any, mid: int) -> None:
        """Set sent values to the columns on index == mid
        like:
            df.loc[mid, col] = value

        Args:
            df (pd.DataFrame): a pandas dataframe object.
            columns (np.ndarray): the dataframe columns that will be receive the values.
            values (Any): values to be inserted
            mid (int): the match_id that will be used as index.
        """
        for col, value in zip(columns, values):
            self.logger.debug(f'setting "{value}" to column "{col}" on match_id {mid}')

            lst_idx = df.last_valid_index()
            lst_idx = lst_idx+1 if lst_idx is not None else 0
            df.loc[mid, col] = value

    def find_waiting(self, by: str, value: str, condition: Callable=EC.presence_of_element_located, timeout: int=30, ignored_exceptions: list[Exception]=None) -> T:
        """Uses WebDriverWait object to find elements

        Args:
            by (str): selenium.webdriver.common.by.By object
            value (str): value of DOM element search
            condition (Callable, optional): callable from selenium.webdriver.support.expected_conditions . Defaults to EC.presence_of_element_located.
            timeout (int, optional): max timeout to wait. Defaults to 30.
            ignored_exceptions (List[Exception], optional): exceptions to ignore. Defaults to None.
        
        Ex:
        >>> button = find_waiting(
            'xpath', 
            '//*[@id="onetrust-close-btn-container"]/button', 
            timeout=10,
            condition=EC.element_to_be_clickable
        )
        >>> elements = find_waiting(
            'xpath', 
            '//*[@id="onetrust-close-btn-container"]/button', 
            timeout=10,
            condition=EC.presence_of_all_elements_located
        )
        """
        wait = WebDriverWait(
            self.browser, 
            timeout=timeout,
            ignored_exceptions=ignored_exceptions
        )
        element = wait.until(condition((by, value)))
        return element
    
    def close_cookie_request(self) -> None:
        """Close the cookies popup
        """
        self.browser.execute_script(
            'arguments[0].click();',
            self.find_waiting('xpath', '//*[@id="onetrust-close-btn-container"]/button')
        )
        self.logger.info('cookies request closed')

    def get_match_links(self) -> None:
        """Get the the link of all matches appending to self.all_match_links.
        """
        MAX_NON_DATA = 2
        non_data_count = 0
        p = 1
        while True:
            self.logger.info(f'getting cards from page {p}')
            sleep(5)

            count_gt_2 = non_data_count > MAX_NON_DATA
            self.logger.debug(f'non_data_count: {non_data_count} - non_data_count > {MAX_NON_DATA} = {count_gt_2}')
            if count_gt_2:
                break
            
            elements = self.find_waiting(
                'css selector', 
                'div.vbw-gs2-match-data-card', 
                EC.presence_of_all_elements_located
            )
            self.logger.debug(f'elements found {len(elements)}')
            try:
                match_links = []
                for card in elements:
                    match_gender = card.find_element(By.CLASS_NAME, 'vbw-gs2-match-gender').text
                    if match_gender.lower() != self.gender:
                        continue

                    link = card.find_element('css selector', 'a').get_attribute('href')
                    match_links.append(link)

        
                self.logger.debug(f'{self.gender} links: {len(match_links)}')

            except StaleElementReferenceException as e:
                self.logger.info(str(e))
                continue

            if match_links:
                self.all_match_links += match_links
                non_data_count = 0
            
            else:
                non_data_count += 1

            self.logger.debug(f'all_match_links count: {len(self.all_match_links)}')
            
            next_page = self.find_waiting(
                'xpath', 
                '/html/body/div[1]/main/section/div[3]/div/div/div[1]/div[2]/div[2]/div/div',
            )
            self.browser.execute_script('arguments[0].click();', next_page)
            self.logger.info('going next page')

            p+=1
        
        self.logger.info(f'{len(self.all_match_links)} matches found')

    def parse_matches(self) -> None:
        """Get the stats from the match and players making self.stats_dataframe 
        and self.players_dataframe
        """
        for mid, match_url in enumerate(self.all_match_links, self.match_id):
            self.browser.get(match_url)
            sleep(3)
            
            pool, phase, matchN = self.find_waiting(
                'xpath', 
                '//*[@id="main-content"]/section/div/div/div[2]/div/a[1]/div[2]/div[1]'
            ).text.split(' - ')

            pool = pool.replace('Pool ', '')
            matchN = re.sub('\D+', '', matchN)
            arena = re.sub(
                r'\n', 
                '-', 
                self.find_waiting('xpath', '//*[@id="main-content"]/section/div/div/div[2]/div/a[1]/div[2]/div[3]').text
            )
            self.logger.debug(f'pool={pool} - phase={phase} - matchN={matchN} - arena={arena}')
            
            home_team = self.find_waiting('xpath', '//*[@id="main-content"]/section/div/div/div[2]/div/a[2]/div[1]/div[1]/div[2]').get_attribute('innerHTML')
            away_team = self.find_waiting('xpath', '//*[@id="main-content"]/section/div/div/div[2]/div/a[2]/div[1]/div[3]/div[2]').get_attribute('innerHTML')
            self.logger.debug(f'home_team={home_team} - away_team={away_team}')

            home_res, away_res = self.find_waiting('xpath', '//*[@id="main-content"]/section/div/div/div[2]/div/a[2]/div[1]/div[2]/div[1]').text.split('\n:\n')
            self.logger.debug(f'home_res={home_res} - away_res={away_res}')

            table = self.find_waiting('xpath', '/html/body/div[1]/main/div/section/div/div[1]/div[1]/div[2]/div/table')
            self.logger.debug(f'table element={table}')

            stats_table = pd.read_html(table.get_attribute('outerHTML'))[0]
            self.logger.debug(stats_table.head(1))

            stats_colsA = (stats_table['Match StatsStats'] + ' H').tolist()
            stats_colsB = (stats_table['Match StatsStats'] + ' A').tolist()
            stats_colsA[-1] = stats_colsA[-1]+'2'
            stats_colsB[-1] =  stats_colsB[-1]+'2'

            stats_df_clean = pd.DataFrame(columns=[*stats_colsA, *stats_colsB])

            stats_df_clean.loc[mid, 'match_id'] = mid
            stats_df_clean.loc[mid, 'home'] = home_team
            stats_df_clean.loc[mid, 'away'] = away_team
            stats_df_clean.loc[mid, 'home_res'] = home_res
            stats_df_clean.loc[mid, 'away_res'] = away_res

            stats_df_clean.loc[mid, 'pool'] = pool
            stats_df_clean.loc[mid, 'phase'] = phase
            stats_df_clean.loc[mid, 'matchN'] = matchN
            stats_df_clean.loc[mid, 'arena'] = arena
            self.set_row(stats_df_clean, stats_colsA, stats_table.iloc[:,1], mid)
            self.set_row(stats_df_clean, stats_colsB, stats_table.iloc[:,3], mid)

            self.matches_dataframe = pd.concat([self.matches_dataframe, stats_df_clean], axis=0)
            self.logger.info(f'stats_dataframe from match_id {mid} concatenated')
            self.logger.debug(self.matches_dataframe.head(1))

            # parte de players
            self.browser.execute_script(
                'arguments[0].click();', 
                self.find_waiting('xpath', '/html/body/div[1]/main/div/section/div/div[1]/div[1]/div[1]/ul/li[2]/a')
            )

            box_tables = self.find_waiting(
                'xpath', 
                '//*[@id="main-content"]/div/section/div/div[1]/div[1]/div[3]/div/div[2]/div/div[2]/div/div/table',
                EC.presence_of_all_elements_located
            )
            box_dfs = []
            for t in box_tables:
                box_dfs.append(
                    pd.read_html(t.get_attribute('outerHTML'))[0]
                )
            self.logger.info(f'{len(box_dfs)} box_dfs found')
            
            piv = len(box_dfs)//2
            dfs_A = box_dfs[:piv]
            dfs_B = box_dfs[piv:]

            cat = np.array(
                [
                    'SCORING',
                    'ATTACK',
                    'BLOCK',
                    'SERVE',
                    'RECEPTION',
                    'DIG',
                    'SET',
                ], 
                dtype=np.object_
            )

            df_a_join = pd.DataFrame()
            df_b_join = pd.DataFrame()
            for i in range(7):
                df_a, df_b = dfs_A[i], dfs_B[i]
                cols_a, cols_b = df_a.columns + f'_{cat[i]}_H', df_b.columns + f'_{cat[i]}_A'

                df_a.columns, df_b.columns = cols_a, cols_b
                df_a, df_b = df_a.set_index(cols_a[1]), df_b.set_index(cols_b[1])
                
                if i == 0:
                    df_a_join = df_a
                    df_b_join = df_b
                else:
                    df_a_join = df_a_join.join(df_a, how='inner')
                    df_b_join = df_b_join.join(df_b, how='inner')
            
            df_a_join['team_H'] = home_team
            df_b_join['team_A'] = away_team

            self.logger.info(f'dfs from {home_team} x {away_team} complete')

            self.players_df_conc = pd.concat([df_a_join.reset_index(inplace=False), df_b_join.reset_index(inplace=False)], axis=1)
            self.players_df_conc['match_id'] = mid
            self.players_dataframe = pd.concat([self.players_dataframe, self.players_df_conc])
            self.logger.info('players_dataframe concatenated')
            self.logger.info(self.players_dataframe.head(1))
    
    def rename_columns(self) -> None:
        """rename the dataframes column if they have the same length
        """
        dfp_cols_n = self.players_dataframe.columns.shape[0]
        pcols_n = len(self.players_cols)

        dfm_cols_n = self.matches_dataframe.columns.shape[0]
        mcols_n = len(self.match_cols)

        values = [
            ['players_dataframe', 'players_cols', dfp_cols_n, pcols_n],
            ['matches_dataframe', 'match_cols', dfm_cols_n, mcols_n]
        ]
        for dfname, colsname, dfl, colsl in values:
            if not dfl != colsl:
                self.logger.warning(f"can't rename {dfname} df {dfl} != {colsl}")
                continue
            
            getattr(self, dfname).columns = getattr(self, colsname)
            self.logger.info(f'columns from {dfname} renamed with success')

    def drop_columns(self) -> None:
        """drop unnecessary columns from dataframes
        """
        pcols = self.players_dataframe.columns[self.players_dataframe.columns.str.startswith('Position')]
        pcols_todrop = pcols[pcols.str.endswith('_H')][1:].tolist() + pcols[pcols.str.endswith('_A')][1:].tolist()

        pcols_todrop = [
            'Player No_ATTACK_H',
            'Player No_BLOCK_H',
            'Player No_SERVE_H',
            'Player No_RECEPTION_H',
            'Player No_DIG_H',
            'Player No_SET_H',
            'Player No_ATTACK_A',
            'Player No_BLOCK_A',
            'Player No_SERVE_A',
            'Player No_RECEPTION_A',
            'Player No_DIG_A',
            'Player No_SET_A',
            *pcols_todrop
        ]
        self.players_dataframe.drop(columns=pcols_todrop, inplace=True)

        mcols_todrop = ['Match Skills H', 'Match Skills A']
        self.matches_dataframe.drop(columns=mcols_todrop, inplace=True)

    def fill_missings(self) -> None:
        for mid in range(self.players_dataframe.match_id.min(), self.players_dataframe.match_id.max()+1):
            mask = self.players_dataframe.match_id == mid
            dfi = self.players_dataframe.loc[mask][['players_H', 'players_A', 'team_H', 'team_A']]
            team_H = dfi['team_H'].loc[~dfi['team_H'].isna()].iloc[0]
            team_A = dfi['team_A'].loc[~dfi['team_A'].isna()].iloc[0]

            self.players_dataframe.loc[(self.players_dataframe['team_H'].isna() & mask), 'team_H'] = team_H
            self.players_dataframe.loc[(self.players_dataframe['team_A'].isna() & mask), 'team_A'] = team_A

            obj_cols = [col for col in self.players_dataframe.columns if self.players_dataframe[col].dtype == 'object']
            self.players_dataframe.loc[mask, obj_cols] = self.players_dataframe.loc[mask, obj_cols].fillna('N/A')
        
        self.players_dataframe.fillna({'Player No_H': -1, 'Player No_A': -1}, inplace=True)
        self.players_dataframe.dropna(axis=1, how='all', inplace=True)
        self.players_dataframe.fillna(0, inplace=True)

    def cast_cols(self):
        col_cast_type = {
            col: np.int32 
            for col in self.players_dataframe.columns 
            if self.players_dataframe[col].dtype == 'float64' and not col.startswith('Efficiency ')
        }
        self.players_dataframe = self.players_dataframe.astype(col_cast_type, errors='ignore')

    def save_dfs(self) -> bool:
        """Save the DataFrame from scraped data to excel format;

        Returns:
            bool: True if saved with success else False.
        """
        try:
            self.matches_dataframe.to_excel(self.stats_dataframe_name, index=False)
            self.logger.info(f'{self.stats_dataframe_name} saved')

            self.players_dataframe.to_excel(self.players_dataframe_name, index=False)
            self.logger.info(f'{self.players_dataframe_name} saved')
            return True
        
        except Exception as e:
            self.logger.error(str(e))
            return False

    @classmethod
    def multiple_years(cls, years: list[int], dates: list[str], **kwargs):
        """Scrape data for multiple years.

        Args:
            years (list[int]): list of the years
            dates (list[str]): list of initial dates
        
        Ex:
        >>> Scraper.multiple_years(years=[2021, 2022], dates=['2021-05-25', '2022-05-31'])
        """
        for year, date in zip(years, dates):
            self = cls(year=year, from_date=date, **kwargs)
            self.run()
