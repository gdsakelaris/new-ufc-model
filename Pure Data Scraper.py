import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
import re
import threading
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class UFCScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("UFC Data Scraper")
        self.root.geometry("850x600")  # Reduced for better visibility
        self.root.minsize(700, 550)  # Set minimum size
        
        self.output_file = tk.StringVar(value="scraped_data.csv")
        self.is_scraping = False
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title with UFC red color
        title_frame = tk.Frame(self.root, bg='Gold')
        title_frame.pack(fill=tk.X)
        tk.Label(title_frame, text="UFC DATA SCRAPER", 
                font=('Arial', 16, 'bold'), fg='white', bg='Gold').pack(pady=(10, 8))
        
        # Instructions
        instr_frame = ttk.LabelFrame(self.root, text="Instructions", padding="5")
        instr_frame.pack(fill=tk.X, padx=10, pady=3)
        
        instructions = """Enter UFC event URLs (one per line) from ufcstats.com. Example: http://www.ufcstats.com/event-details/5efaaf313b652dd7
Find URLs at: http://ufcstats.com/statistics/events/completed"""
        
        ttk.Label(instr_frame, text=instructions, font=('Arial', 8), justify=tk.LEFT, foreground="gray").pack(anchor=tk.W)
        
        # URL input
        url_frame = ttk.LabelFrame(self.root, text="Event URLs", padding="5")
        url_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=3)
        
        self.url_text = scrolledtext.ScrolledText(url_frame, height=6, width=95)
        self.url_text.pack(fill=tk.BOTH, expand=True)
        
        # Output file
        output_frame = ttk.LabelFrame(self.root, text="Output File", padding="5")
        output_frame.pack(fill=tk.X, padx=10, pady=3)
        
        ttk.Entry(output_frame, textvariable=self.output_file, width=60).pack(side=tk.LEFT, padx=5)
        ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(side=tk.LEFT)
        
        # Progress area
        progress_frame = ttk.LabelFrame(self.root, text="Progress", padding="5")
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=3)
        
        self.progress_text = scrolledtext.ScrolledText(progress_frame, height=8, width=95, state='disabled')
        self.progress_text.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        button_frame = ttk.Frame(self.root, padding="5")
        button_frame.pack(fill=tk.X, padx=10)
        
        ttk.Button(button_frame, text="Clear", command=self.clear_urls).pack(side=tk.LEFT, padx=3)
        self.scrape_btn = ttk.Button(button_frame, text="Start Scraping", 
                                     command=self.start_scraping)
        self.scrape_btn.pack(side=tk.RIGHT, padx=3)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save Scraped Data As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )
        if filename:
            self.output_file.set(filename)
            
    def clear_urls(self):
        self.url_text.delete('1.0', tk.END)
        
    def log_progress(self, message):
        self.progress_text.config(state='normal')
        self.progress_text.insert(tk.END, f"{message}\n")
        self.progress_text.see(tk.END)
        self.progress_text.config(state='disabled')
        self.root.update()
        
    def start_scraping(self):
        if self.is_scraping:
            return
            
        urls_text = self.url_text.get('1.0', tk.END).strip()
        if not urls_text:
            messagebox.showwarning("No URLs", "Please enter at least one event URL")
            return
            
        event_urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        
        if not event_urls:
            messagebox.showwarning("No URLs", "Please enter valid event URLs")
            return
            
        self.is_scraping = True
        self.scrape_btn.config(state='disabled', text='Scraping...')
        self.status_var.set("Scraping in progress...")
        self.progress_text.config(state='normal')
        self.progress_text.delete('1.0', tk.END)
        self.progress_text.config(state='disabled')
        
        thread = threading.Thread(target=self.run_scraper, args=(event_urls,))
        thread.daemon = True
        thread.start()
        
    def run_scraper(self, event_urls):
        try:
            self.log_progress(f"Starting scrape of {len(event_urls)} event(s)...")
            
            df = self.create_large_dataset(event_urls)
            
            if df is not None and not df.empty:
                # Drop data leakage columns (current stats that leak future info)
                leakage_cols = [
                    'r_current_age', 'b_current_age', 'current_age_diff',
                    'r_wins', 'b_wins', 'wins_diff',
                    'r_losses', 'b_losses', 'losses_diff',
                    'r_draws', 'b_draws', 'draws_diff',
                    'r_win_loss_ratio', 'b_win_loss_ratio', 'win_loss_ratio_diff',
                    'r_pro_SLpM', 'b_pro_SLpM', 'pro_SLpM_diff',
                    'r_pro_sig_str_acc', 'b_pro_sig_str_acc', 'pro_sig_str_acc_diff',
                    'r_pro_SApM', 'b_pro_SApM', 'pro_SApM_diff',
                    'r_pro_str_def', 'b_pro_str_def', 'pro_str_def_diff',
                    'r_pro_td_avg', 'b_pro_td_avg', 'pro_td_avg_diff',
                    'r_pro_td_acc', 'b_pro_td_acc', 'pro_td_acc_diff',
                    'r_pro_td_def', 'b_pro_td_def', 'pro_td_def_diff',
                    'r_pro_sub_avg', 'b_pro_sub_avg', 'pro_sub_avg_diff',
                ]
                df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)

                output_path = self.output_file.get()
                if output_path.endswith('.xlsx'):
                    df.to_excel(output_path, index=False)
                else:
                    df.to_csv(output_path, index=False, float_format="%.2f")
                    
                self.log_progress(f"\n✓ Success! Complete dataset saved to {output_path}")
                self.log_progress(f"✓ Total fights: {len(df)}")
                self.log_progress(f"✓ Total columns: {len(df.columns)}")
                self.status_var.set(f"Complete! {len(df)} fights with {len(df.columns)} columns")
                messagebox.showinfo("Success", f"Scraped data for {len(df)} fights!\n\nSaved to: {output_path}")
            else:
                self.log_progress("\n✗ No data was scraped")
                self.status_var.set("No data scraped")
                messagebox.showwarning("No Data", "No fight data was found")
                
        except Exception as e:
            self.log_progress(f"\n✗ Error: {str(e)}")
            self.status_var.set("Error occurred")
            messagebox.showerror("Error", f"An error occurred:\n\n{str(e)}")
            
        finally:
            self.is_scraping = False
            self.scrape_btn.config(state='normal', text='Start Scraping')
            
    # ========== COMPREHENSIVE SCRAPING FUNCTIONS FROM DATA_SCRAPER_2.PY ==========
    
    def parse_height(self, height_str):
        if not height_str or height_str == "--":
            return None
        try:
            height_str = height_str.replace("Height:", "").strip()
            match = re.search(r"(\d+)'?\s*(\d+)\"?", height_str)
            if match:
                feet, inches = map(int, match.groups())
                return round((feet * 12 + inches), 2)
            match = re.search(r"(\d+)\"?", height_str)
            if match:
                return round(int(match.group(1)), 2)
            return None
        except:
            return None
            
    def parse_weight(self, weight_str):
        if weight_str in ["--", "", None]:
            return None
        try:
            match = re.search(r"(\d+) lbs.", weight_str)
            if match:
                return round(int(match.group(1)), 2)
            return None
        except:
            return None
            
    def parse_reach(self, reach_str):
        if reach_str in ["--", "", None] or "Reach:--" in str(reach_str):
            return None
        try:
            reach = reach_str.replace("Reach:", "").replace('"', "").strip()
            return round(float(reach), 2)
        except:
            return None
            
    def parse_dob(self, dob_str):
        if dob_str in ["--", "", None]:
            return None
        try:
            dob = dob_str.replace("DOB:", "").strip()
            return datetime.strptime(dob, "%b %d, %Y").date()
        except:
            return None
            
    def safe_float(self, value):
        if value in ["--", "---", "", None]:
            return None
        try:
            return float(value)
        except:
            return None
            
    def clean_weight_class(self, weight_class):
        male_weights = ["Flyweight", "Bantamweight", "Featherweight", "Lightweight", 
                       "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight", "Catch Weight"]
        female_weights = ["Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", 
                         "Women's Featherweight", "Women's Catch Weight"]
        all_weights = male_weights + female_weights
        
        cleaned_input = str(weight_class).lower()
        for weight in all_weights:
            pattern = r"\b" + re.escape(weight.lower()) + r"\b"
            if re.search(pattern, cleaned_input):
                return weight
        return weight_class.strip()
        
    def get_event_info(self, event_url):
        try:
            response = requests.get(event_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            event_name = soup.find('h2', class_='b-content__title').text.strip()
            event_date = None
            event_location = None
            
            for item in soup.find_all('li', class_='b-list__box-list-item'):
                if 'Date:' in item.text:
                    date_text = item.text.split('Date:')[-1].strip()
                    try:
                        event_date = datetime.strptime(date_text, '%B %d, %Y').date()
                    except:
                        pass
                elif 'Location:' in item.text:
                    event_location = item.text.split('Location:')[-1].strip()
                    
            return {'event_name': event_name, 'event_url': event_url, 'event_date': event_date, 'event_location': event_location}
        except Exception as e:
            self.log_progress(f"  Error getting event info: {str(e)}")
            return None
            
    def get_fight_urls(self, event_urls):
        fight_urls = []
        event_info = []
        
        for i, url in enumerate(event_urls):
            self.log_progress(f"Fetching event {i+1}/{len(event_urls)}: {url}")
            event_data = self.get_event_info(url)
            
            if event_data:
                try:
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    for row in soup.find_all('tr', class_='b-fight-details__table-row'):
                        link = row.find('a', class_='b-flag b-flag_style_green')
                        if not link:
                            link = row.find('a', class_=lambda x: x and 'b-flag' in x)
                            if link and 'draw' not in link.get_text(strip=True).lower():
                                link = None
                        if link:
                            fight_urls.append(link.get('href'))
                            event_info.append(event_data)

                    self.log_progress(f"  Found {len([e for e in event_info if e == event_data])} completed fights")
                except Exception as e:
                    self.log_progress(f"  Error: {str(e)}")
                    
        return fight_urls, event_info
        
    def get_fighter_stats(self, fighter_url):
        try:
            response = requests.get(fighter_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            fighter_name = soup.find('span', class_='b-content__title-highlight').text.strip()
            
            record = soup.find('span', class_='b-content__title-record').text.replace('Record:', '').strip()
            record_parts = record.split('-')
            wins = int(record_parts[0])
            losses = int(record_parts[1])
            draws = 0
            if len(record_parts) > 2:
                if 'NC' in record_parts[2]:
                    nc_part = record_parts[2].split()
                    draws = int(nc_part[0]) if len(nc_part) > 1 else 0
                else:
                    draws = int(record_parts[2])
                    
            stats = [s.get_text(strip=True) for s in soup.find_all('li', class_='b-list__box-list-item b-list__box-list-item_type_block')]
            
            height = self.parse_height(stats[0])
            weight = self.parse_weight(stats[1])
            reach = self.parse_reach(stats[2])
            stance = stats[3].replace('STANCE:', '').strip()
            dob = self.parse_dob(stats[4])
            
            slpm = self.safe_float(stats[5].replace('SLpM:', '').strip())
            str_acc = self.safe_float(stats[6].replace('Str. Acc.:', '').rstrip('%'))
            str_acc = str_acc / 100 if str_acc else None
            sapm = self.safe_float(stats[7].replace('SApM:', '').strip())
            str_def = self.safe_float(stats[8].replace('Str. Def:', '').rstrip('%'))
            str_def = str_def / 100 if str_def else None
            td_avg = self.safe_float(stats[10].replace('TD Avg.:', '').strip())
            td_acc = self.safe_float(stats[11].replace('TD Acc.:', '').rstrip('%'))
            td_acc = td_acc / 100 if td_acc else None
            td_def = self.safe_float(stats[12].replace('TD Def.:', '').rstrip('%'))
            td_def = td_def / 100 if td_def else None
            sub_avg = self.safe_float(stats[13].replace('Sub. Avg.:', '').strip())
            
            win_loss_ratio = wins / max(losses, 1)
            ape_index = reach / height if height and reach and height > 0 else None
            
            return {
                'name': fighter_name, 'wins': wins, 'losses': losses, 'draws': draws,
                'height': height, 'weight': weight, 'reach': reach, 'stance': stance,
                'date_of_birth': dob, 'pro_SLpM': slpm, 'pro_sig_str_acc': str_acc,
                'pro_SApM': sapm, 'pro_str_def': str_def, 'pro_td_avg': td_avg,
                'pro_td_acc': td_acc, 'pro_td_def': td_def, 'pro_sub_avg': sub_avg,
                'win_loss_ratio': win_loss_ratio, 'ape_index': ape_index
            }
        except Exception as e:
            self.log_progress(f"    Error getting fighter stats: {str(e)}")
            return None
            
    def scrape_fight_details(self, fight_url, event_data):
        try:
            response = requests.get(fight_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get fighters
            fighters = [f.get_text(strip=True) for f in soup.find_all('h3', class_='b-fight-details__person-name')]
            fighter_statuses = [s.get_text(strip=True) for s in soup.find_all('i', class_='b-fight-details__person-status')]
            if fighter_statuses[0] == 'W':
                winner = 'Red'
            elif fighter_statuses[1] == 'W':
                winner = 'Blue'
            else:
                winner = 'Draw'

            # Fight details
            fight_title = soup.find('i', class_='b-fight-details__fight-title')
            fight_title = fight_title.text.strip() if fight_title else None
            weight_class = self.clean_weight_class(fight_title.split(' Bout')[0].strip()) if fight_title else None

            method = soup.find('i', class_='b-fight-details__text-item_first')
            method = method.text.replace('Method:', '').strip() if method else None

            if winner == 'Draw':
                method = 'Draw'
            elif any(x in method for x in ["Doctor's Stoppage", 'DQ']):
                method = 'KO/TKO'
                
            gen_stats = [s.get_text(strip=True) for s in soup.find_all('i', class_='b-fight-details__text-item')]
            
            is_title_bout = 1 if 'Title' in fight_title else 0
            gender = 'Women' if "Women's" in fight_title else 'Men'
            total_rounds = int(re.search(r'(\d+)', gen_stats[2].replace('Time format:', '')).group(1)) if len(gen_stats) > 2 and re.search(r'(\d+)', gen_stats[2].replace('Time format:', '')) else 3
            finish_round = int(gen_stats[0].replace('Round:', '')) if len(gen_stats) > 0 else 1
            
            time_parts = gen_stats[1].replace('Time:', '').split(':') if len(gen_stats) > 1 else ['0', '0']
            minutes, seconds = map(int, time_parts)
            time_sec = minutes * 60 + seconds
            total_fight_time_sec = (finish_round - 1) * 300 + time_sec
            
            referee = next((item.split(':')[1].strip() for item in gen_stats if 'Referee:' in item), None)
            
            # Fight stats
            stats_data = [s.text.strip() for s in soup.find_all('p', class_='b-fight-details__table-text')]
            
            def parse_ratio(ratio_str):
                if ratio_str in ['---', '--', '']:
                    return 0, 0
                parts = ratio_str.split(' of ')
                return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
                
            def parse_control_time(time_str):
                if time_str in ['---', '--', '', None]:
                    return 0
                try:
                    minutes, seconds = map(int, time_str.split(':'))
                    return minutes * 60 + seconds
                except:
                    return 0
                    
            r_sig_str, r_sig_str_att = parse_ratio(stats_data[4]) if len(stats_data) > 4 else (0, 0)
            b_sig_str, b_sig_str_att = parse_ratio(stats_data[5]) if len(stats_data) > 5 else (0, 0)
            r_total_str, r_total_str_att = parse_ratio(stats_data[8]) if len(stats_data) > 8 else (0, 0)
            b_total_str, b_total_str_att = parse_ratio(stats_data[9]) if len(stats_data) > 9 else (0, 0)
            r_td, r_td_att = parse_ratio(stats_data[10]) if len(stats_data) > 10 else (0, 0)
            b_td, b_td_att = parse_ratio(stats_data[11]) if len(stats_data) > 11 else (0, 0)
            
            r_kd = int(stats_data[2]) if len(stats_data) > 2 and stats_data[2] not in ['---', '--', ''] else 0
            b_kd = int(stats_data[3]) if len(stats_data) > 3 and stats_data[3] not in ['---', '--', ''] else 0
            r_sub_att = int(stats_data[14]) if len(stats_data) > 14 and stats_data[14] not in ['---', '--', ''] else 0
            b_sub_att = int(stats_data[15]) if len(stats_data) > 15 and stats_data[15] not in ['---', '--', ''] else 0
            r_rev = int(stats_data[16]) if len(stats_data) > 16 and stats_data[16] not in ['---', '--', ''] else 0
            b_rev = int(stats_data[17]) if len(stats_data) > 17 and stats_data[17] not in ['---', '--', ''] else 0
            r_ctrl_sec = parse_control_time(stats_data[18]) if len(stats_data) > 18 else 0
            b_ctrl_sec = parse_control_time(stats_data[19]) if len(stats_data) > 19 else 0
            
            # Strike breakdown
            strike_percentages = {}
            for chart in soup.find_all('div', class_='b-fight-details__charts-row'):
                title_elem = chart.find('i', class_='b-fight-details__charts-row-title')
                if title_elem:
                    title = title_elem.text.strip().lower()
                    red_val = chart.find('i', class_=lambda x: x and 'red' in x)
                    blue_val = chart.find('i', class_=lambda x: x and 'blue' in x)

                    if red_val and blue_val and title in ['head', 'body', 'leg', 'distance', 'clinch', 'ground']:
                        strike_percentages[f'r_{title}'] = round(float(red_val.text.strip().rstrip('%')) / 100, 2)
                        strike_percentages[f'b_{title}'] = round(float(blue_val.text.strip().rstrip('%')) / 100, 2)

            # Per-round stats
            # Uses the same flat list of all <p class="b-fight-details__table-text"> elements
            # as the existing totals parsing, accessed by calculated offset.
            #
            # Page layout (each cell has 2 p elements, one per fighter):
            #   [0..19]   : totals table (10 cols × 2)
            #   [20..20+20*R-1] : per-round totals (20 items × R rounds)
            #   [20+20*R..20+20*R+17] : sig strikes totals (9 cols × 2)
            #   [20+20*R+18..] : sig strikes per-round (18 items × R rounds)
            # where R = finish_round (rounds actually fought)
            round_stats = {}
            actual_rounds = finish_round
            sig_rnd_base = 20 + 20 * actual_rounds + 18

            def sd(idx):
                return stats_data[idx] if idx < len(stats_data) else ''

            def safe_int(v):
                return int(v) if v not in ('---', '--', '') else 0

            for n in range(1, actual_rounds + 1):
                p = f'rd{n}'
                t = 20 + 20 * (n - 1)   # base index in per-round totals
                s = sig_rnd_base + 18 * (n - 1)  # base index in per-round sig strikes

                # Totals: KD, sig str, total str, TD, sub att, rev, ctrl
                round_stats[f'r_{p}_kd'] = safe_int(sd(t+2))
                round_stats[f'b_{p}_kd'] = safe_int(sd(t+3))

                r_l, r_a = parse_ratio(sd(t+4))
                b_l, b_a = parse_ratio(sd(t+5))
                round_stats[f'r_{p}_sig_str'] = r_l
                round_stats[f'r_{p}_sig_str_att'] = r_a
                round_stats[f'b_{p}_sig_str'] = b_l
                round_stats[f'b_{p}_sig_str_att'] = b_a

                r_l, r_a = parse_ratio(sd(t+8))
                b_l, b_a = parse_ratio(sd(t+9))
                round_stats[f'r_{p}_str'] = r_l
                round_stats[f'r_{p}_str_att'] = r_a
                round_stats[f'b_{p}_str'] = b_l
                round_stats[f'b_{p}_str_att'] = b_a

                r_l, r_a = parse_ratio(sd(t+10))
                b_l, b_a = parse_ratio(sd(t+11))
                round_stats[f'r_{p}_td'] = r_l
                round_stats[f'r_{p}_td_att'] = r_a
                round_stats[f'b_{p}_td'] = b_l
                round_stats[f'b_{p}_td_att'] = b_a

                round_stats[f'r_{p}_sub_att'] = safe_int(sd(t+14))
                round_stats[f'b_{p}_sub_att'] = safe_int(sd(t+15))
                round_stats[f'r_{p}_rev'] = safe_int(sd(t+16))
                round_stats[f'b_{p}_rev'] = safe_int(sd(t+17))
                round_stats[f'r_{p}_ctrl_sec'] = parse_control_time(sd(t+18))
                round_stats[f'b_{p}_ctrl_sec'] = parse_control_time(sd(t+19))

                # Sig strikes: head, body, leg, distance, clinch, ground
                r_l, r_a = parse_ratio(sd(s+6))
                b_l, b_a = parse_ratio(sd(s+7))
                round_stats[f'r_{p}_head'] = r_l
                round_stats[f'r_{p}_head_att'] = r_a
                round_stats[f'b_{p}_head'] = b_l
                round_stats[f'b_{p}_head_att'] = b_a

                r_l, r_a = parse_ratio(sd(s+8))
                b_l, b_a = parse_ratio(sd(s+9))
                round_stats[f'r_{p}_body'] = r_l
                round_stats[f'r_{p}_body_att'] = r_a
                round_stats[f'b_{p}_body'] = b_l
                round_stats[f'b_{p}_body_att'] = b_a

                r_l, r_a = parse_ratio(sd(s+10))
                b_l, b_a = parse_ratio(sd(s+11))
                round_stats[f'r_{p}_leg'] = r_l
                round_stats[f'r_{p}_leg_att'] = r_a
                round_stats[f'b_{p}_leg'] = b_l
                round_stats[f'b_{p}_leg_att'] = b_a

                r_l, r_a = parse_ratio(sd(s+12))
                b_l, b_a = parse_ratio(sd(s+13))
                round_stats[f'r_{p}_distance'] = r_l
                round_stats[f'r_{p}_distance_att'] = r_a
                round_stats[f'b_{p}_distance'] = b_l
                round_stats[f'b_{p}_distance_att'] = b_a

                r_l, r_a = parse_ratio(sd(s+14))
                b_l, b_a = parse_ratio(sd(s+15))
                round_stats[f'r_{p}_clinch'] = r_l
                round_stats[f'r_{p}_clinch_att'] = r_a
                round_stats[f'b_{p}_clinch'] = b_l
                round_stats[f'b_{p}_clinch_att'] = b_a

                r_l, r_a = parse_ratio(sd(s+16))
                b_l, b_a = parse_ratio(sd(s+17))
                round_stats[f'r_{p}_ground'] = r_l
                round_stats[f'r_{p}_ground_att'] = r_a
                round_stats[f'b_{p}_ground'] = b_l
                round_stats[f'b_{p}_ground_att'] = b_a

            # Fill in empty values for rounds that weren't fought (up to 5)
            for n in range(actual_rounds + 1, 6):
                p = f'rd{n}'
                for prefix in ('r_', 'b_'):
                    round_stats[f'{prefix}{p}_kd'] = ''
                    round_stats[f'{prefix}{p}_sig_str'] = ''
                    round_stats[f'{prefix}{p}_sig_str_att'] = ''
                    round_stats[f'{prefix}{p}_str'] = ''
                    round_stats[f'{prefix}{p}_str_att'] = ''
                    round_stats[f'{prefix}{p}_td'] = ''
                    round_stats[f'{prefix}{p}_td_att'] = ''
                    round_stats[f'{prefix}{p}_sub_att'] = ''
                    round_stats[f'{prefix}{p}_rev'] = ''
                    round_stats[f'{prefix}{p}_ctrl_sec'] = ''
                    round_stats[f'{prefix}{p}_head'] = ''
                    round_stats[f'{prefix}{p}_head_att'] = ''
                    round_stats[f'{prefix}{p}_body'] = ''
                    round_stats[f'{prefix}{p}_body_att'] = ''
                    round_stats[f'{prefix}{p}_leg'] = ''
                    round_stats[f'{prefix}{p}_leg_att'] = ''
                    round_stats[f'{prefix}{p}_distance'] = ''
                    round_stats[f'{prefix}{p}_distance_att'] = ''
                    round_stats[f'{prefix}{p}_clinch'] = ''
                    round_stats[f'{prefix}{p}_clinch_att'] = ''
                    round_stats[f'{prefix}{p}_ground'] = ''
                    round_stats[f'{prefix}{p}_ground_att'] = ''

            fight_data = {
                'event_name': event_data['event_name'],
                'event_url': event_data['event_url'],
                'fight_url': fight_url,
                'event_date': event_data['event_date'],
                'event_location': event_data['event_location'],
                'r_fighter': fighters[0],
                'b_fighter': fighters[1],
                'winner': winner,
                'weight_class': weight_class,
                'is_title_bout': is_title_bout,
                'gender': gender,
                'method': method,
                'finish_round': finish_round,
                'total_rounds': total_rounds,
                'time_sec': time_sec,
                'total_fight_time_sec': total_fight_time_sec,
                'referee': referee,
                'r_kd': r_kd, 'b_kd': b_kd,
                'r_sig_str': r_sig_str, 'r_sig_str_att': r_sig_str_att,
                'b_sig_str': b_sig_str, 'b_sig_str_att': b_sig_str_att,
                'r_str': r_total_str, 'r_str_att': r_total_str_att,
                'b_str': b_total_str, 'b_str_att': b_total_str_att,
                'r_td': r_td, 'r_td_att': r_td_att,
                'b_td': b_td, 'b_td_att': b_td_att,
                'r_sub_att': r_sub_att, 'b_sub_att': b_sub_att,
                'r_rev': r_rev, 'b_rev': b_rev,
                'r_ctrl_sec': r_ctrl_sec, 'b_ctrl_sec': b_ctrl_sec,
                **strike_percentages,
                **round_stats
            }
            
            return fight_data
        except Exception as e:
            self.log_progress(f"    Error scraping fight: {str(e)}")
            return None
            
    def create_large_dataset(self, event_urls):
        # Get all fight URLs
        fight_urls, event_info = self.get_fight_urls(event_urls)
        self.log_progress(f"\nTotal fights to scrape: {len(fight_urls)}\n")
        
        # Get all fighter URLs
        self.log_progress("Collecting fighter URLs...")
        fighter_urls = []
        for i, url in enumerate(fight_urls):
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', class_='b-link b-fight-details__person-link'):
                    fighter_urls.append(link.get('href'))
            except:
                pass
                
        self.log_progress(f"Found {len(fighter_urls)} fighter profiles to scrape\n")
        
        # Get fighter stats
        self.log_progress("Scraping fighter profiles...")
        fighters_stats = []
        for i, url in enumerate(fighter_urls):
            stats = self.get_fighter_stats(url)
            if stats:
                fighters_stats.append(stats)
            if (i + 1) % 10 == 0:
                self.log_progress(f"  Processed {i + 1}/{len(fighter_urls)} fighters")
                
        # Split into red/blue
        red_fighters = [fighters_stats[i] for i in range(len(fighters_stats)) if i % 2 == 0]
        blue_fighters = [fighters_stats[i] for i in range(len(fighters_stats)) if i % 2 != 0]
        
        self.log_progress(f"\nScraping fight details...")
        all_fights = []
        
        for i, (fight_url, event_data) in enumerate(zip(fight_urls, event_info)):
            self.log_progress(f"  Fight {i + 1}/{len(fight_urls)}")
            
            fight_data = self.scrape_fight_details(fight_url, event_data)
            if fight_data and i < len(red_fighters) and i < len(blue_fighters):
                # Add fighter stats
                for key, val in red_fighters[i].items():
                    fight_data[f'r_{key}'] = val
                for key, val in blue_fighters[i].items():
                    fight_data[f'b_{key}'] = val
                    
                all_fights.append(fight_data)
                
        if not all_fights:
            return None
            
        df = pd.DataFrame(all_fights)
        
        # Calculate accuracy percentages
        df['r_sig_str_acc'] = df.apply(lambda r: round(r['r_sig_str'] / r['r_sig_str_att'], 2) if r['r_sig_str_att'] > 0 else 0, axis=1)
        df['b_sig_str_acc'] = df.apply(lambda r: round(r['b_sig_str'] / r['b_sig_str_att'], 2) if r['b_sig_str_att'] > 0 else 0, axis=1)
        df['r_str_acc'] = df.apply(lambda r: round(r['r_str'] / r['r_str_att'], 2) if r['r_str_att'] > 0 else 0, axis=1)
        df['b_str_acc'] = df.apply(lambda r: round(r['b_str'] / r['b_str_att'], 2) if r['b_str_att'] > 0 else 0, axis=1)
        df['r_td_acc'] = df.apply(lambda r: round(r['r_td'] / r['r_td_att'], 2) if r['r_td_att'] > 0 else 0, axis=1)
        df['b_td_acc'] = df.apply(lambda r: round(r['b_td'] / r['b_td_att'], 2) if r['b_td_att'] > 0 else 0, axis=1)
        
        # Calculate ages
        def calc_age(born, event_date):
            if pd.isna(born) or pd.isna(event_date):
                return None
            try:
                return event_date.year - born.year - ((event_date.month, event_date.day) < (born.month, born.day))
            except:
                return None
                
        df['r_age_at_event'] = df.apply(lambda r: calc_age(r.get('r_date_of_birth'), r.get('event_date')), axis=1)
        df['b_age_at_event'] = df.apply(lambda r: calc_age(r.get('b_date_of_birth'), r.get('event_date')), axis=1)
        # For historical fights use age_at_event; for upcoming fights (no event_date) use today
        df['r_current_age'] = df.apply(
            lambda r: calc_age(r.get('r_date_of_birth'), r.get('event_date'))
            if pd.notna(r.get('event_date'))
            else calc_age(r.get('r_date_of_birth'), datetime.now().date()),
            axis=1,
        )
        df['b_current_age'] = df.apply(
            lambda r: calc_age(r.get('b_date_of_birth'), r.get('event_date'))
            if pd.notna(r.get('event_date'))
            else calc_age(r.get('b_date_of_birth'), datetime.now().date()),
            axis=1,
        )
        
        # Calculate differences
        diff_cols = ['wins', 'losses', 'draws', 'height', 'weight', 'reach', 'age_at_event', 'current_age',
                     'kd', 'sig_str', 'sig_str_att', 'sig_str_acc', 'str', 'str_att', 'str_acc',
                     'td', 'td_att', 'td_acc', 'sub_att', 'rev', 'ctrl_sec', 'win_loss_ratio', 'ape_index',
                     'head', 'body', 'leg', 'distance', 'clinch', 'ground']
        
        for col in diff_cols:
            if f'r_{col}' in df.columns and f'b_{col}' in df.columns:
                df[f'{col}_diff'] = (pd.to_numeric(df[f'r_{col}'], errors='coerce') - 
                                    pd.to_numeric(df[f'b_{col}'], errors='coerce')).round(2)
                                    
        pro_stats = ['pro_SLpM', 'pro_SApM', 'pro_str_def', 'pro_td_avg', 'pro_td_acc', 'pro_td_def', 'pro_sub_avg', 'pro_sig_str_acc']
        for stat in pro_stats:
            if f'r_{stat}' in df.columns and f'b_{stat}' in df.columns:
                df[f'{stat}_diff'] = (pd.to_numeric(df[f'r_{stat}'], errors='coerce') - 
                                     pd.to_numeric(df[f'b_{stat}'], errors='coerce')).round(2)
        
        # Format dates
        if 'event_date' in df.columns:
            df['event_date'] = pd.to_datetime(df['event_date']).dt.strftime('%m/%d/%Y')
        if 'r_date_of_birth' in df.columns:
            df['r_date_of_birth'] = df['r_date_of_birth'].astype(str)
        if 'b_date_of_birth' in df.columns:
            df['b_date_of_birth'] = df['b_date_of_birth'].astype(str)
        
        # Reorder columns to match original scraper
        column_order = [
            'event_date', 'event_name', 'event_url', 'fight_url', 'event_location', 'r_fighter', 'b_fighter',
            'weight_class', 'gender', 'total_rounds', 'is_title_bout', 'referee',
            'winner', 'method', 'finish_round', 'time_sec', 'total_fight_time_sec',
            'r_height', 'b_height', 'height_diff',
            'r_reach', 'b_reach', 'reach_diff',
            'r_ape_index', 'b_ape_index', 'ape_index_diff',
            'r_stance', 'b_stance',
            'r_weight', 'b_weight', 'weight_diff',
            'r_date_of_birth', 'b_date_of_birth',
            'r_age_at_event', 'b_age_at_event', 'age_at_event_diff',
            'r_current_age', 'b_current_age', 'current_age_diff',
            'r_wins', 'b_wins', 'wins_diff',
            'r_losses', 'b_losses', 'losses_diff',
            'r_draws', 'b_draws', 'draws_diff',
            'r_win_loss_ratio', 'b_win_loss_ratio', 'win_loss_ratio_diff',
            'r_sig_str', 'b_sig_str', 'sig_str_diff',
            'r_sig_str_att', 'b_sig_str_att', 'sig_str_att_diff',
            'r_sig_str_acc', 'b_sig_str_acc', 'sig_str_acc_diff',
            'r_str', 'b_str', 'str_diff',
            'r_str_att', 'b_str_att', 'str_att_diff',
            'r_str_acc', 'b_str_acc', 'str_acc_diff',
            'r_kd', 'b_kd', 'kd_diff',
            'r_head', 'b_head', 'head_diff',
            'r_body', 'b_body', 'body_diff',
            'r_leg', 'b_leg', 'leg_diff',
            'r_distance', 'b_distance', 'distance_diff',
            'r_clinch', 'b_clinch', 'clinch_diff',
            'r_ground', 'b_ground', 'ground_diff',
            'r_td', 'b_td', 'td_diff',
            'r_td_att', 'b_td_att', 'td_att_diff',
            'r_td_acc', 'b_td_acc', 'td_acc_diff',
            'r_sub_att', 'b_sub_att', 'sub_att_diff',
            'r_rev', 'b_rev', 'rev_diff',
            'r_ctrl_sec', 'b_ctrl_sec', 'ctrl_sec_diff',
            'r_pro_SLpM', 'b_pro_SLpM', 'pro_SLpM_diff',
            'r_pro_sig_str_acc', 'b_pro_sig_str_acc', 'pro_sig_str_acc_diff',
            'r_pro_SApM', 'b_pro_SApM', 'pro_SApM_diff',
            'r_pro_str_def', 'b_pro_str_def', 'pro_str_def_diff',
            'r_pro_td_avg', 'b_pro_td_avg', 'pro_td_avg_diff',
            'r_pro_td_acc', 'b_pro_td_acc', 'pro_td_acc_diff',
            'r_pro_td_def', 'b_pro_td_def', 'pro_td_def_diff',
            'r_pro_sub_avg', 'b_pro_sub_avg', 'pro_sub_avg_diff',
            'r_name', 'b_name',
        ]

        # Add per-round columns for rounds 1-5
        for n in range(1, 6):
            p = f'rd{n}'
            column_order.extend([
                f'r_{p}_kd', f'b_{p}_kd',
                f'r_{p}_sig_str', f'r_{p}_sig_str_att', f'b_{p}_sig_str', f'b_{p}_sig_str_att',
                f'r_{p}_str', f'r_{p}_str_att', f'b_{p}_str', f'b_{p}_str_att',
                f'r_{p}_td', f'r_{p}_td_att', f'b_{p}_td', f'b_{p}_td_att',
                f'r_{p}_sub_att', f'b_{p}_sub_att',
                f'r_{p}_rev', f'b_{p}_rev',
                f'r_{p}_ctrl_sec', f'b_{p}_ctrl_sec',
                f'r_{p}_head', f'r_{p}_head_att', f'b_{p}_head', f'b_{p}_head_att',
                f'r_{p}_body', f'r_{p}_body_att', f'b_{p}_body', f'b_{p}_body_att',
                f'r_{p}_leg', f'r_{p}_leg_att', f'b_{p}_leg', f'b_{p}_leg_att',
                f'r_{p}_distance', f'r_{p}_distance_att', f'b_{p}_distance', f'b_{p}_distance_att',
                f'r_{p}_clinch', f'r_{p}_clinch_att', f'b_{p}_clinch', f'b_{p}_clinch_att',
                f'r_{p}_ground', f'r_{p}_ground_att', f'b_{p}_ground', f'b_{p}_ground_att',
            ])
        
        # Keep only columns that exist in the dataframe, in the specified order
        existing_columns = [col for col in column_order if col in df.columns]
        # Add any columns not in column_order at the end
        remaining_columns = [col for col in df.columns if col not in column_order]
        final_order = existing_columns + remaining_columns
        
        df = df[final_order]
            
        return df

def main():
    root = tk.Tk()
    app = UFCScraperApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()