# UFC Fighter Comparison GUI Dashboard
"""
UFC Fighter Comparison GUI Dashboard
==================================================

This script provides a GUI interface for comparing UFC fighters in batch mode with
comprehensive career statistics. Output is saved as an Excel workbook with separate
sheets for each matchup.

Features:
- Searchable fighter selection with autocomplete dropdown
- Batch mode - add multiple matchups before generating
- Excel output with separate sheets per matchup
- Comprehensive statistics display

Usage:
    python "Fighter Comparison Dashboard.py"

Author: AI Assistant
Date: 2025
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For Excel output
try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("Warning: openpyxl not installed. Install with: pip install openpyxl")

class FighterComparisonGUI:
    def __init__(self, root, data_file=None):
        self.root = root
        self.root.title("UFC Fighter Comparison Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')

        if data_file is None:
            data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fight_data.csv')

        # Load fight data
        print("Loading fight data...")
        try:
            self.df = pd.read_csv(data_file)
            self.df['event_date'] = pd.to_datetime(self.df['event_date'])
            self.df = self.df.sort_values('event_date')
            print(f"Loaded {len(self.df)} fights from {self.df['event_date'].min().strftime('%Y-%m-%d')} to {self.df['event_date'].max().strftime('%Y-%m-%d')}")
        except FileNotFoundError:
            messagebox.showerror("Error", f"Could not find '{data_file}' file.")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {e}")
            return

        # Get unique fighter names
        self.all_fighters = sorted(set(self.df['r_fighter'].tolist() + self.df['b_fighter'].tolist()))

        # Batch matchups list
        self.matchups = []

        self.create_widgets()

    def create_widgets(self):
        """Create the GUI widgets"""
        # Main title
        title_label = tk.Label(
            self.root,
            text="UFC FIGHTER COMPARISON DASHBOARD - BATCH MODE",
            font=('Arial', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=10)

        # Search frame
        search_frame = tk.Frame(self.root, bg='#2c3e50')
        search_frame.pack(pady=20)

        # Fighter 1 search
        tk.Label(search_frame, text="Fighter 1:", font=('Arial', 12, 'bold'), bg='#2c3e50', fg='white').grid(row=0, column=0, padx=10, pady=5)
        self.fighter1_var = tk.StringVar()
        self.fighter1_combo = ttk.Combobox(
            search_frame,
            textvariable=self.fighter1_var,
            font=('Arial', 11),
            width=30
        )
        self.fighter1_combo.grid(row=0, column=1, padx=10, pady=5)
        self.fighter1_combo['values'] = self.all_fighters
        self.fighter1_combo.bind('<KeyRelease>', lambda e: self.filter_fighters(1))

        # Fighter 2 search
        tk.Label(search_frame, text="Fighter 2:", font=('Arial', 12, 'bold'), bg='#2c3e50', fg='white').grid(row=0, column=2, padx=10, pady=5)
        self.fighter2_var = tk.StringVar()
        self.fighter2_combo = ttk.Combobox(
            search_frame,
            textvariable=self.fighter2_var,
            font=('Arial', 11),
            width=30
        )
        self.fighter2_combo.grid(row=0, column=3, padx=10, pady=5)
        self.fighter2_combo['values'] = self.all_fighters
        self.fighter2_combo.bind('<KeyRelease>', lambda e: self.filter_fighters(2))

        # Add Matchup button
        add_btn = tk.Button(
            search_frame,
            text="Add Matchup",
            command=self.add_matchup,
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=5
        )
        add_btn.grid(row=0, column=4, padx=10, pady=5)

        # Generate Excel button
        generate_btn = tk.Button(
            search_frame,
            text="Generate Excel",
            command=self.generate_excel,
            font=('Arial', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            padx=20,
            pady=5
        )
        generate_btn.grid(row=0, column=5, padx=10, pady=5)

        # Clear All button
        clear_btn = tk.Button(
            search_frame,
            text="Clear All",
            command=self.clear_matchups,
            font=('Arial', 10),
            bg='#95a5a6',
            fg='white',
            padx=10,
            pady=5
        )
        clear_btn.grid(row=0, column=6, padx=10, pady=5)

        # Paste fights section
        paste_frame = tk.LabelFrame(
            self.root,
            text="PASTE FIGHTS (CSV)",
            font=('Arial', 11, 'bold'),
            bg='#2c3e50',
            fg='white',
            padx=10,
            pady=5
        )
        paste_frame.pack(pady=(0, 5), padx=20, fill=tk.X)

        paste_hint = tk.Label(
            paste_frame,
            text="Format: Fighter1,Fighter2[,WeightClass,Gender,Rounds]  — one fight per line",
            font=('Arial', 9),
            bg='#2c3e50',
            fg='#bdc3c7'
        )
        paste_hint.pack(anchor='w')

        paste_inner = tk.Frame(paste_frame, bg='#2c3e50')
        paste_inner.pack(fill=tk.X, pady=5)

        paste_scroll = tk.Scrollbar(paste_inner, orient=tk.VERTICAL)
        paste_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.paste_text = tk.Text(
            paste_inner,
            font=('Consolas', 10),
            bg='#34495e',
            fg='white',
            insertbackground='white',
            height=5,
            yscrollcommand=paste_scroll.set,
            wrap=tk.NONE
        )
        self.paste_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        paste_scroll.config(command=self.paste_text.yview)

        parse_btn = tk.Button(
            paste_frame,
            text="Parse & Add All",
            command=self.parse_and_add_fights,
            font=('Arial', 11, 'bold'),
            bg='#8e44ad',
            fg='white',
            padx=15,
            pady=4
        )
        parse_btn.pack(anchor='e', pady=(4, 0))

        # Matchups list frame
        matchups_label = tk.Label(
            self.root,
            text="MATCHUPS QUEUE:",
            font=('Arial', 14, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        matchups_label.pack(pady=(10, 5))

        # Matchups listbox with scrollbar
        list_frame = tk.Frame(self.root, bg='#2c3e50')
        list_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.matchups_listbox = tk.Listbox(
            list_frame,
            font=('Consolas', 11),
            bg='#34495e',
            fg='white',
            selectmode=tk.SINGLE,
            yscrollcommand=scrollbar.set,
            height=15
        )
        self.matchups_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.matchups_listbox.yview)

        # Bind double-click to remove matchup
        self.matchups_listbox.bind('<Double-Button-1>', lambda e: self.remove_selected_matchup())

        # Remove button
        remove_btn = tk.Button(
            self.root,
            text="Remove Selected Matchup",
            command=self.remove_selected_matchup,
            font=('Arial', 10),
            bg='#e67e22',
            fg='white',
            padx=10,
            pady=5
        )
        remove_btn.pack(pady=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Add matchups to the queue and generate Excel workbook")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=('Arial', 10),
            bg='#2c3e50',
            fg='#bdc3c7',
            anchor='w'
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)

    def filter_fighters(self, fighter_num):
        """Filter fighters based on search input"""
        search_term = self.fighter1_var.get() if fighter_num == 1 else self.fighter2_var.get()

        if len(search_term) < 2:
            combo = self.fighter1_combo if fighter_num == 1 else self.fighter2_combo
            combo['values'] = []
            return

        # Filter fighters that contain the search term
        filtered_fighters = [fighter for fighter in self.all_fighters
                           if search_term.lower() in fighter.lower()]

        combo = self.fighter1_combo if fighter_num == 1 else self.fighter2_combo
        combo['values'] = filtered_fighters[:20]  # Limit to 20 results

    def add_matchup(self):
        """Add a matchup to the queue"""
        fighter1 = self.fighter1_var.get().strip()
        fighter2 = self.fighter2_var.get().strip()

        if not fighter1 or not fighter2:
            messagebox.showwarning("Warning", "Please select both fighters before adding matchup.")
            return

        if fighter1 not in self.all_fighters:
            messagebox.showwarning("Warning", f"Fighter '{fighter1}' not found in database.")
            return

        if fighter2 not in self.all_fighters:
            messagebox.showwarning("Warning", f"Fighter '{fighter2}' not found in database.")
            return

        # Add to matchups list
        matchup = {'fighter1': fighter1, 'fighter2': fighter2, 'weight_class': None, 'gender': None, 'rounds': None}
        self.matchups.append(matchup)

        # Update listbox
        self.matchups_listbox.insert(tk.END, f"{len(self.matchups)}. {fighter1} vs {fighter2}")

        # Clear selections
        self.fighter1_var.set('')
        self.fighter2_var.set('')

        self.status_var.set(f"Added matchup: {fighter1} vs {fighter2} ({len(self.matchups)} total)")

    def remove_selected_matchup(self):
        """Remove selected matchup from queue"""
        selection = self.matchups_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a matchup to remove.")
            return

        index = selection[0]
        removed = self.matchups.pop(index)

        # Refresh listbox
        self.matchups_listbox.delete(0, tk.END)
        for i, m in enumerate(self.matchups, 1):
            display = f"{i}. {m['fighter1']} vs {m['fighter2']}"
            if m.get('weight_class'):
                display += f" ({m['weight_class']})"
            self.matchups_listbox.insert(tk.END, display)

        self.status_var.set(f"Removed matchup: {removed['fighter1']} vs {removed['fighter2']}")

    def clear_matchups(self):
        """Clear all matchups"""
        if not self.matchups:
            return

        if messagebox.askyesno("Confirm", "Clear all matchups?"):
            self.matchups.clear()
            self.matchups_listbox.delete(0, tk.END)
            self.status_var.set("All matchups cleared")

    def parse_and_add_fights(self):
        """Parse pasted CSV fight data and add to queue"""
        text = self.paste_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please paste fight data first.")
            return

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        added = 0
        skipped = []

        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                skipped.append(f"'{line}' (need at least 2 fields)")
                continue

            fighter1 = parts[0]
            fighter2 = parts[1]
            weight_class = parts[2] if len(parts) > 2 else None
            gender = parts[3] if len(parts) > 3 else None
            try:
                rounds = int(parts[4]) if len(parts) > 4 else None
            except ValueError:
                rounds = None

            if fighter1 not in self.all_fighters:
                skipped.append(f"'{fighter1}' not in database")
                continue
            if fighter2 not in self.all_fighters:
                skipped.append(f"'{fighter2}' not in database")
                continue

            matchup = {'fighter1': fighter1, 'fighter2': fighter2,
                       'weight_class': weight_class, 'gender': gender, 'rounds': rounds}
            self.matchups.append(matchup)

            display = f"{len(self.matchups)}. {fighter1} vs {fighter2}"
            if weight_class:
                display += f" ({weight_class})"
            self.matchups_listbox.insert(tk.END, display)
            added += 1

        msg = f"Added {added} matchup(s) from paste"
        if skipped:
            preview = ', '.join(skipped[:5])
            if len(skipped) > 5:
                preview += f" and {len(skipped) - 5} more"
            msg += f"\nSkipped: {preview}"
            messagebox.showwarning("Partial Import", msg)
        self.status_var.set(msg.splitlines()[0])

        # Clear paste area on success
        if added > 0:
            self.paste_text.delete("1.0", tk.END)

    def generate_excel(self):
        """Generate Excel workbook with all matchups"""
        if not self.matchups:
            messagebox.showwarning("Warning", "No matchups in queue. Please add matchups first.")
            return

        if not HAS_OPENPYXL:
            messagebox.showerror("Error", "openpyxl not installed. Install with: pip install openpyxl")
            return

        # Ask for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialfile="UFC_Comparisons.xlsx"
        )

        if not filename:
            return

        try:
            self.status_var.set("Generating Excel workbook...")
            self.root.update()

            # Create workbook
            wb = Workbook()
            wb.remove(wb.active)  # Remove default sheet

            # Process each matchup
            for i, matchup in enumerate(self.matchups, 1):
                fighter1 = matchup['fighter1']
                fighter2 = matchup['fighter2']
                self.status_var.set(f"Processing matchup {i}/{len(self.matchups)}: {fighter1} vs {fighter2}")
                self.root.update()

                # Get fighter data
                fighter1_stats = self.get_fighter_data(fighter1)
                fighter2_stats = self.get_fighter_data(fighter2)

                if not fighter1_stats or not fighter2_stats:
                    messagebox.showwarning("Warning", f"Could not find data for matchup: {fighter1} vs {fighter2}")
                    continue

                # Create sheet for this matchup
                sheet_name = f"{fighter1[:15]} vs {fighter2[:15]}"[:31]  # Excel sheet name limit
                ws = wb.create_sheet(title=sheet_name)

                # Generate comparison data in sheet
                self.write_comparison_to_sheet(ws, fighter1_stats, fighter2_stats)

            # Save workbook
            wb.save(filename)

            self.status_var.set(f"Excel workbook saved: {filename}")
            messagebox.showinfo("Success", f"Generated {len(wb.sheetnames)} comparison sheets!\n\nSaved to: {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Error generating Excel: {e}")
            self.status_var.set("Error occurred")

    def write_comparison_to_sheet(self, ws, fighter1_stats, fighter2_stats):
        """Write comparison data to Excel sheet"""
        current_row = 1

        # Title
        ws.merge_cells(f'A{current_row}:C{current_row}')
        title_cell = ws[f'A{current_row}']
        title_cell.value = "UFC FIGHTER COMPARISON"
        title_cell.font = Font(size=16, bold=True, color="FFFFFF")
        title_cell.alignment = Alignment(horizontal='left')
        title_cell.fill = PatternFill(start_color="2c3e50", end_color="2c3e50", fill_type="solid")
        current_row += 2

        # Helper function to add section
        def add_section(title, data, start_row):
            # Section header
            ws.merge_cells(f'A{start_row}:C{start_row}')
            header_cell = ws[f'A{start_row}']
            header_cell.value = title
            header_cell.font = Font(size=12, bold=True, color="FFFFFF")
            header_cell.fill = PatternFill(start_color="34495e", end_color="34495e", fill_type="solid")
            header_cell.alignment = Alignment(horizontal='left')
            start_row += 1

            # Column headers
            ws[f'A{start_row}'] = 'Metric'
            ws[f'B{start_row}'] = fighter1_stats['name']
            ws[f'C{start_row}'] = fighter2_stats['name']
            for col in ['A', 'B', 'C']:
                cell = ws[f'{col}{start_row}']
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="bdc3c7", end_color="bdc3c7", fill_type="solid")
                cell.alignment = Alignment(horizontal='left')
            start_row += 1

            # Data rows
            for row_data in data:
                ws[f'A{start_row}'] = row_data[0]
                ws[f'B{start_row}'] = row_data[1]
                ws[f'C{start_row}'] = row_data[2]
                # Set left alignment for all data cells
                for col in ['A', 'B', 'C']:
                    ws[f'{col}{start_row}'].alignment = Alignment(horizontal='left')
                start_row += 1

            return start_row + 1

        # BASIC INFORMATION
        basic_data = [
            ['Name', fighter1_stats['name'], fighter2_stats['name']],
            ['Total Fights', fighter1_stats['total_fights'], fighter2_stats['total_fights']],
            ['Height', fighter1_stats['height'], fighter2_stats['height']],
            ['Reach', fighter1_stats['reach'], fighter2_stats['reach']],
            ['Stance', fighter1_stats['stance'], fighter2_stats['stance']],
            ['Age', fighter1_stats['current_age'], fighter2_stats['current_age']],
            ['Weight', fighter1_stats['weight'], fighter2_stats['weight']]
        ]
        current_row = add_section('BASIC INFORMATION', basic_data, current_row)

        # FIGHT RECORD
        record_data = [
            ['Total Fights', fighter1_stats['total_fights'], fighter2_stats['total_fights']],
            ['Wins', fighter1_stats['wins'], fighter2_stats['wins']],
            ['Losses', fighter1_stats['losses'], fighter2_stats['losses']],
            ['Win Rate', f"{fighter1_stats['win_rate']*100:.1f}%", f"{fighter2_stats['win_rate']*100:.1f}%"]
        ]
        current_row = add_section('FIGHT RECORD', record_data, current_row)

        # WINS & LOSSES BY METHOD
        method_data = [
            ['KO/TKO', f"{fighter1_stats['wins_by_method']['KO/TKO']}-{fighter1_stats['losses_by_method']['KO/TKO']}",
             f"{fighter2_stats['wins_by_method']['KO/TKO']}-{fighter2_stats['losses_by_method']['KO/TKO']}"],
            ['Submission', f"{fighter1_stats['wins_by_method']['Submission']}-{fighter1_stats['losses_by_method']['Submission']}",
             f"{fighter2_stats['wins_by_method']['Submission']}-{fighter2_stats['losses_by_method']['Submission']}"],
            ['Decision', f"{fighter1_stats['wins_by_method']['Decision']}-{fighter1_stats['losses_by_method']['Decision']}",
             f"{fighter2_stats['wins_by_method']['Decision']}-{fighter2_stats['losses_by_method']['Decision']}"]
        ]
        current_row = add_section('WINS & LOSSES BY METHOD (W-L)', method_data, current_row)

        # CAREER AVERAGES
        # Format fight time as min:sec
        f1_time_min = int(fighter1_stats['avg_fight_time_seconds'] // 60)
        f1_time_sec = int(fighter1_stats['avg_fight_time_seconds'] % 60)
        f2_time_min = int(fighter2_stats['avg_fight_time_seconds'] // 60)
        f2_time_sec = int(fighter2_stats['avg_fight_time_seconds'] % 60)

        career_data = [
            ['Avg Fight Time', f"{f1_time_min}:{f1_time_sec:02d}", f"{f2_time_min}:{f2_time_sec:02d}"],
            ['SLpM (Sig Strikes/Min)', f"{fighter1_stats['career_averages']['slpm']:.2f}", f"{fighter2_stats['career_averages']['slpm']:.2f}"],
            ['Sig Strike Accuracy (%)', f"{fighter1_stats['career_averages']['sig_str_accuracy']*100:.1f}%", f"{fighter2_stats['career_averages']['sig_str_accuracy']*100:.1f}%"],
            ['SApM (Sig Absorbed/Min)', f"{fighter1_stats['career_averages']['sapm']:.2f}", f"{fighter2_stats['career_averages']['sapm']:.2f}"],
            ['Strike Defense (%)', f"{fighter1_stats['career_averages']['str_defense']*100:.1f}%", f"{fighter2_stats['career_averages']['str_defense']*100:.1f}%"],
            ['TD per 15min', f"{fighter1_stats['career_averages']['td_avg_per_15min']:.2f}", f"{fighter2_stats['career_averages']['td_avg_per_15min']:.2f}"],
            ['TD Accuracy (%)', f"{fighter1_stats['career_averages']['td_accuracy']*100:.1f}%", f"{fighter2_stats['career_averages']['td_accuracy']*100:.1f}%"],
            ['TD Defense (%)', f"{fighter1_stats['career_averages']['td_defense']*100:.1f}%", f"{fighter2_stats['career_averages']['td_defense']*100:.1f}%"],
            ['Sub per 15min', f"{fighter1_stats['career_averages']['sub_avg_per_15min']:.2f}", f"{fighter2_stats['career_averages']['sub_avg_per_15min']:.2f}"]
        ]
        current_row = add_section('CAREER AVERAGES', career_data, current_row)

        # STRIKE DISTRIBUTION (Landed/Absorbed combined)
        strike_dist_data = [
            ['Head (Avg Landed/Absorbed)',
             f"{fighter1_stats['strike_distribution_avg_per_fight']['head_avg']:.1f} / {fighter1_stats['strike_distribution_absorbed_avg_per_fight']['head_absorbed_avg']:.1f}",
             f"{fighter2_stats['strike_distribution_avg_per_fight']['head_avg']:.1f} / {fighter2_stats['strike_distribution_absorbed_avg_per_fight']['head_absorbed_avg']:.1f}"],
            ['Body (Avg Landed/Absorbed)',
             f"{fighter1_stats['strike_distribution_avg_per_fight']['body_avg']:.1f} / {fighter1_stats['strike_distribution_absorbed_avg_per_fight']['body_absorbed_avg']:.1f}",
             f"{fighter2_stats['strike_distribution_avg_per_fight']['body_avg']:.1f} / {fighter2_stats['strike_distribution_absorbed_avg_per_fight']['body_absorbed_avg']:.1f}"],
            ['Leg (Avg Landed/Absorbed)',
             f"{fighter1_stats['strike_distribution_avg_per_fight']['leg_avg']:.1f} / {fighter1_stats['strike_distribution_absorbed_avg_per_fight']['leg_absorbed_avg']:.1f}",
             f"{fighter2_stats['strike_distribution_avg_per_fight']['leg_avg']:.1f} / {fighter2_stats['strike_distribution_absorbed_avg_per_fight']['leg_absorbed_avg']:.1f}"],
            ['Distance (Avg Landed/Absorbed)',
             f"{fighter1_stats['strike_distribution_avg_per_fight']['distance_avg']:.1f} / {fighter1_stats['strike_distribution_absorbed_avg_per_fight']['distance_absorbed_avg']:.1f}",
             f"{fighter2_stats['strike_distribution_avg_per_fight']['distance_avg']:.1f} / {fighter2_stats['strike_distribution_absorbed_avg_per_fight']['distance_absorbed_avg']:.1f}"],
            ['Clinch (Avg Landed/Absorbed)',
             f"{fighter1_stats['strike_distribution_avg_per_fight']['clinch_avg']:.1f} / {fighter1_stats['strike_distribution_absorbed_avg_per_fight']['clinch_absorbed_avg']:.1f}",
             f"{fighter2_stats['strike_distribution_avg_per_fight']['clinch_avg']:.1f} / {fighter2_stats['strike_distribution_absorbed_avg_per_fight']['clinch_absorbed_avg']:.1f}"],
            ['Ground (Avg Landed/Absorbed)',
             f"{fighter1_stats['strike_distribution_avg_per_fight']['ground_avg']:.1f} / {fighter1_stats['strike_distribution_absorbed_avg_per_fight']['ground_absorbed_avg']:.1f}",
             f"{fighter2_stats['strike_distribution_avg_per_fight']['ground_avg']:.1f} / {fighter2_stats['strike_distribution_absorbed_avg_per_fight']['ground_absorbed_avg']:.1f}"]
        ]
        current_row = add_section('STRIKE DISTRIBUTION (PER FIGHT)', strike_dist_data, current_row)

        # GRAPPLING STATISTICS
        grappling_data = [
            ['Takedowns (Avg Landed/Absorbed)',
             f"{fighter1_stats['avg_takedowns_per_fight']:.2f} / {fighter1_stats['avg_takedowns_against_per_fight']:.2f}",
             f"{fighter2_stats['avg_takedowns_per_fight']:.2f} / {fighter2_stats['avg_takedowns_against_per_fight']:.2f}"],
            ['Submissions (Avg Landed/Absorbed)',
             f"{fighter1_stats['avg_submission_attempts_per_fight']:.2f} / {fighter1_stats['avg_submission_attempts_against_per_fight']:.2f}",
             f"{fighter2_stats['avg_submission_attempts_per_fight']:.2f} / {fighter2_stats['avg_submission_attempts_against_per_fight']:.2f}"],
            ['Reversals (Avg Landed/Absorbed)',
             f"{fighter1_stats['avg_reversals_per_fight']:.2f} / {fighter1_stats['avg_reversals_against_per_fight']:.2f}",
             f"{fighter2_stats['avg_reversals_per_fight']:.2f} / {fighter2_stats['avg_reversals_against_per_fight']:.2f}"],
            ['Control Time (Avg For/Against sec)',
             f"{fighter1_stats['avg_control_time_per_fight']:.1f} / {fighter1_stats['avg_control_time_against_per_fight']:.1f}",
             f"{fighter2_stats['avg_control_time_per_fight']:.1f} / {fighter2_stats['avg_control_time_against_per_fight']:.1f}"]
        ]
        current_row = add_section('GRAPPLING STATISTICS (PER FIGHT)', grappling_data, current_row)

        # KNOCKDOWN STATISTICS (with totals)
        knockdown_data = [
            ['Total Landed', f"{fighter1_stats['total_knockdowns_landed']:.0f}", f"{fighter2_stats['total_knockdowns_landed']:.0f}"],
            ['Avg Landed', f"{fighter1_stats['avg_knockdowns_per_fight']:.2f}", f"{fighter2_stats['avg_knockdowns_per_fight']:.2f}"],
            ['Total Against', f"{fighter1_stats['total_knockdowns_against']:.0f}", f"{fighter2_stats['total_knockdowns_against']:.0f}"],
            ['Avg Against', f"{fighter1_stats['avg_knockdowns_against_per_fight']:.2f}", f"{fighter2_stats['avg_knockdowns_against_per_fight']:.2f}"]
        ]
        current_row = add_section('KNOCKDOWN STATISTICS', knockdown_data, current_row)

        # Add spacer
        current_row += 1

        # Helper function to add fight history
        def add_fight_history(fighter_name, fight_history, start_row):
            # Section header
            ws.merge_cells(f'A{start_row}:E{start_row}')
            header_cell = ws[f'A{start_row}']
            header_cell.value = f"UFC FIGHT HISTORY - {fighter_name}"
            header_cell.font = Font(size=12, bold=True, color="FFFFFF")
            header_cell.fill = PatternFill(start_color="34495e", end_color="34495e", fill_type="solid")
            header_cell.alignment = Alignment(horizontal='left')
            start_row += 1

            # Column headers
            ws[f'A{start_row}'] = 'Date'
            ws[f'B{start_row}'] = 'Opponent'
            ws[f'C{start_row}'] = 'Result'
            ws[f'D{start_row}'] = 'Method'
            ws[f'E{start_row}'] = 'Round'
            for col in ['A', 'B', 'C', 'D', 'E']:
                cell = ws[f'{col}{start_row}']
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="bdc3c7", end_color="bdc3c7", fill_type="solid")
                cell.alignment = Alignment(horizontal='left')
            start_row += 1

            # Data rows - reverse order (most recent first)
            for fight in reversed(fight_history):
                date_str = fight['date'].strftime('%Y-%m-%d') if pd.notna(fight['date']) else 'N/A'
                ws[f'A{start_row}'] = date_str
                ws[f'B{start_row}'] = fight['opponent']
                ws[f'C{start_row}'] = fight['result']
                ws[f'D{start_row}'] = fight['method']
                ws[f'E{start_row}'] = f"R{fight['round']}"

                # Set left alignment for all cells
                for col in ['A', 'B', 'C', 'D', 'E']:
                    ws[f'{col}{start_row}'].alignment = Alignment(horizontal='left')

                # Color code result
                result_cell = ws[f'C{start_row}']
                if fight['result'] == 'Win':
                    result_cell.fill = PatternFill(start_color="d4edda", end_color="d4edda", fill_type="solid")
                elif fight['result'] == 'Loss':
                    result_cell.fill = PatternFill(start_color="f8d7da", end_color="f8d7da", fill_type="solid")

                start_row += 1

            return start_row + 1

        # Add Fight History for Fighter 1
        current_row = add_fight_history(fighter1_stats['name'], fighter1_stats['fight_history'], current_row)

        # Add spacer
        current_row += 1

        # Add Fight History for Fighter 2
        current_row = add_fight_history(fighter2_stats['name'], fighter2_stats['fight_history'], current_row)

        # Auto-size columns
        for col in ['A', 'B', 'C', 'D', 'E']:
            max_length = 0
            column = ws[col]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except (TypeError, AttributeError):
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[col].width = adjusted_width

    def get_fighter_data(self, fighter_name):
        """Get all fights for a specific fighter and calculate career statistics"""
        # Find all fights where the fighter was either red or blue corner
        red_fights = self.df[self.df['r_fighter'].str.contains(fighter_name, case=False, na=False)]
        blue_fights = self.df[self.df['b_fighter'].str.contains(fighter_name, case=False, na=False)]

        if red_fights.empty and blue_fights.empty:
            return None

        # Combine and sort by date
        fighter_fights = []

        for _, fight in red_fights.iterrows():
            fight_data = {
                'date': fight['event_date'],
                'opponent': fight['b_fighter'],
                'result': 'Win' if fight['winner'] == 'Red' else 'Loss' if fight['winner'] == 'Blue' else 'Draw',
                'method': fight['method'],
                'round': fight['finish_round'],
                'time': fight['time_sec'],
                'total_time': fight['total_fight_time_sec'],
                'is_red': True,
                'height': fight['r_height'],
                'reach': fight['r_reach'],
                'stance': fight['r_stance'],
                'weight': fight['r_weight'],
                'age_at_fight': fight['r_age_at_event'],
                'date_of_birth': fight['r_date_of_birth'],
                # Striking stats
                'sig_str_landed': fight['r_sig_str'],
                'sig_str_attempted': fight['r_sig_str_att'],
                'knockdowns': fight['r_kd'],
                # Strike distribution percentages
                'head_pct': fight['r_head'],
                'body_pct': fight['r_body'],
                'leg_pct': fight['r_leg'],
                'distance_pct': fight['r_distance'],
                'clinch_pct': fight['r_clinch'],
                'ground_pct': fight['r_ground'],
                # Grappling stats
                'takedowns_landed': fight['r_td'],
                'takedowns_attempted': fight['r_td_att'],
                'submission_attempts': fight['r_sub_att'],
                'reversals': fight['r_rev'],
                'control_time': fight['r_ctrl_sec'],
                # Career averages going into this fight
                'career_slpm': fight['r_pro_SLpM'],
                'career_sig_str_acc': fight['r_pro_sig_str_acc'],
                'career_sapm': fight['r_pro_SApM'],
                'career_str_def': fight['r_pro_str_def'],
                'career_td_avg': fight['r_pro_td_avg'],
                'career_td_acc': fight['r_pro_td_acc'],
                'career_td_def': fight['r_pro_td_def'],
                'career_sub_avg': fight['r_pro_sub_avg']
            }
            fighter_fights.append(fight_data)

        for _, fight in blue_fights.iterrows():
            fight_data = {
                'date': fight['event_date'],
                'opponent': fight['r_fighter'],
                'result': 'Win' if fight['winner'] == 'Blue' else 'Loss' if fight['winner'] == 'Red' else 'Draw',
                'method': fight['method'],
                'round': fight['finish_round'],
                'time': fight['time_sec'],
                'total_time': fight['total_fight_time_sec'],
                'is_red': False,
                'height': fight['b_height'],
                'reach': fight['b_reach'],
                'stance': fight['b_stance'],
                'weight': fight['b_weight'],
                'age_at_fight': fight['b_age_at_event'],
                'date_of_birth': fight['b_date_of_birth'],
                # Striking stats
                'sig_str_landed': fight['b_sig_str'],
                'sig_str_attempted': fight['b_sig_str_att'],
                'knockdowns': fight['b_kd'],
                # Strike distribution percentages
                'head_pct': fight['b_head'],
                'body_pct': fight['b_body'],
                'leg_pct': fight['b_leg'],
                'distance_pct': fight['b_distance'],
                'clinch_pct': fight['b_clinch'],
                'ground_pct': fight['b_ground'],
                # Grappling stats
                'takedowns_landed': fight['b_td'],
                'takedowns_attempted': fight['b_td_att'],
                'submission_attempts': fight['b_sub_att'],
                'reversals': fight['b_rev'],
                'control_time': fight['b_ctrl_sec'],
                # Career averages going into this fight
                'career_slpm': fight['b_pro_SLpM'],
                'career_sig_str_acc': fight['b_pro_sig_str_acc'],
                'career_sapm': fight['b_pro_SApM'],
                'career_str_def': fight['b_pro_str_def'],
                'career_td_avg': fight['b_pro_td_avg'],
                'career_td_acc': fight['b_pro_td_acc'],
                'career_td_def': fight['b_pro_td_def'],
                'career_sub_avg': fight['b_pro_sub_avg']
            }
            fighter_fights.append(fight_data)

        # Convert to DataFrame and sort by date
        fighter_df = pd.DataFrame(fighter_fights)
        fighter_df = fighter_df.sort_values('date').reset_index(drop=True)

        if fighter_df.empty:
            return None

        return self.calculate_career_stats(fighter_df, fighter_name)

    def calculate_career_stats(self, fighter_df, fighter_name):
        """Calculate comprehensive career statistics for a fighter"""
        stats = {
            'name': fighter_name,
            'total_fights': len(fighter_df),
            'wins': len(fighter_df[fighter_df['result'] == 'Win']),
            'losses': len(fighter_df[fighter_df['result'] == 'Loss']),
            'draws': len(fighter_df[fighter_df['result'] == 'Draw'])
        }

        # Basic info from most recent fight
        latest_fight = fighter_df.iloc[-1]
        stats['height'] = latest_fight['height']
        stats['reach'] = latest_fight['reach']
        stats['stance'] = latest_fight['stance']
        stats['weight'] = latest_fight['weight']

        # Calculate current age
        if pd.notna(latest_fight['date_of_birth']):
            try:
                dob = pd.to_datetime(latest_fight['date_of_birth'])
                stats['current_age'] = (datetime.now() - dob).days // 365
            except (ValueError, TypeError):
                stats['current_age'] = latest_fight['age_at_fight'] + (datetime.now().year - latest_fight['date'].year)
        else:
            stats['current_age'] = latest_fight['age_at_fight'] + (datetime.now().year - latest_fight['date'].year)

        # Calculate method breakdown for wins and losses (excluding DQ and No Contest)
        wins_df = fighter_df[fighter_df['result'] == 'Win']
        losses_df = fighter_df[fighter_df['result'] == 'Loss']

        def normalize_method(m):
            m = str(m) if pd.notna(m) else ''
            if 'Decision' in m:
                return 'Decision'
            if 'KO/TKO' in m or 'TKO' in m:
                return 'KO/TKO'
            if 'Submission' in m:
                return 'Submission'
            return m

        stats['wins_by_method'] = wins_df['method'].apply(normalize_method).value_counts().to_dict()
        stats['losses_by_method'] = losses_df['method'].apply(normalize_method).value_counts().to_dict()

        # Fill in missing methods with 0 (excluding DQ and No Contest)
        all_methods = ['KO/TKO', 'Submission', 'Decision']
        for method in all_methods:
            if method not in stats['wins_by_method']:
                stats['wins_by_method'][method] = 0
            if method not in stats['losses_by_method']:
                stats['losses_by_method'][method] = 0

        # Calculate averages per fight for strike distribution
        stats['strike_distribution_avg_per_fight'] = {
            'head_avg': (fighter_df['sig_str_landed'] * fighter_df['head_pct']).sum() / stats['total_fights'],
            'body_avg': (fighter_df['sig_str_landed'] * fighter_df['body_pct']).sum() / stats['total_fights'],
            'leg_avg': (fighter_df['sig_str_landed'] * fighter_df['leg_pct']).sum() / stats['total_fights'],
            'distance_avg': (fighter_df['sig_str_landed'] * fighter_df['distance_pct']).sum() / stats['total_fights'],
            'clinch_avg': (fighter_df['sig_str_landed'] * fighter_df['clinch_pct']).sum() / stats['total_fights'],
            'ground_avg': (fighter_df['sig_str_landed'] * fighter_df['ground_pct']).sum() / stats['total_fights']
        }

        # Calculate absorbed strikes by looking at opponent data
        absorbed_dist = {'head_absorbed': 0, 'body_absorbed': 0, 'leg_absorbed': 0,
                        'distance_absorbed': 0, 'clinch_absorbed': 0, 'ground_absorbed': 0}

        for _, fight in fighter_df.iterrows():
            if fight['is_red']:
                opponent_fight = self.df[
                    (self.df['r_fighter'] == fighter_name) &
                    (self.df['b_fighter'] == fight['opponent']) &
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    opp_sig_str = opponent_fight.iloc[0]['b_sig_str']
                    absorbed_dist['head_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['b_head']
                    absorbed_dist['body_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['b_body']
                    absorbed_dist['leg_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['b_leg']
                    absorbed_dist['distance_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['b_distance']
                    absorbed_dist['clinch_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['b_clinch']
                    absorbed_dist['ground_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['b_ground']
            else:
                opponent_fight = self.df[
                    (self.df['b_fighter'] == fighter_name) &
                    (self.df['r_fighter'] == fight['opponent']) &
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    opp_sig_str = opponent_fight.iloc[0]['r_sig_str']
                    absorbed_dist['head_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['r_head']
                    absorbed_dist['body_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['r_body']
                    absorbed_dist['leg_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['r_leg']
                    absorbed_dist['distance_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['r_distance']
                    absorbed_dist['clinch_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['r_clinch']
                    absorbed_dist['ground_absorbed'] += opp_sig_str * opponent_fight.iloc[0]['r_ground']

        stats['strike_distribution_absorbed_avg_per_fight'] = {
            'head_absorbed_avg': absorbed_dist['head_absorbed'] / stats['total_fights'],
            'body_absorbed_avg': absorbed_dist['body_absorbed'] / stats['total_fights'],
            'leg_absorbed_avg': absorbed_dist['leg_absorbed'] / stats['total_fights'],
            'distance_absorbed_avg': absorbed_dist['distance_absorbed'] / stats['total_fights'],
            'clinch_absorbed_avg': absorbed_dist['clinch_absorbed'] / stats['total_fights'],
            'ground_absorbed_avg': absorbed_dist['ground_absorbed'] / stats['total_fights']
        }

        # Career averages (from the data)
        latest_career_avgs = fighter_df.iloc[-1]
        stats['career_averages'] = {
            'slpm': latest_career_avgs['career_slpm'],
            'sig_str_accuracy': latest_career_avgs['career_sig_str_acc'],
            'sapm': latest_career_avgs['career_sapm'],
            'str_defense': latest_career_avgs['career_str_def'],
            'td_avg_per_15min': latest_career_avgs['career_td_avg'],
            'td_accuracy': latest_career_avgs['career_td_acc'],
            'td_defense': latest_career_avgs['career_td_def'],
            'sub_avg_per_15min': latest_career_avgs['career_sub_avg']
        }

        # Grappling averages per fight
        stats['avg_takedowns_per_fight'] = fighter_df['takedowns_landed'].sum() / stats['total_fights']
        stats['avg_submission_attempts_per_fight'] = fighter_df['submission_attempts'].sum() / stats['total_fights']
        stats['avg_reversals_per_fight'] = fighter_df['reversals'].sum() / stats['total_fights']
        stats['avg_control_time_per_fight'] = fighter_df['control_time'].sum() / stats['total_fights']

        # Calculate against stats
        total_takedowns_against = 0
        total_submissions_against = 0
        total_reversals_against = 0
        total_control_time_against = 0

        for _, fight in fighter_df.iterrows():
            if fight['is_red']:
                opponent_fight = self.df[
                    (self.df['r_fighter'] == fighter_name) &
                    (self.df['b_fighter'] == fight['opponent']) &
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    total_takedowns_against += opponent_fight.iloc[0]['b_td']
                    total_submissions_against += opponent_fight.iloc[0]['b_sub_att']
                    total_reversals_against += opponent_fight.iloc[0]['b_rev']
                    total_control_time_against += opponent_fight.iloc[0]['b_ctrl_sec']
            else:
                opponent_fight = self.df[
                    (self.df['b_fighter'] == fighter_name) &
                    (self.df['r_fighter'] == fight['opponent']) &
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    total_takedowns_against += opponent_fight.iloc[0]['r_td']
                    total_submissions_against += opponent_fight.iloc[0]['r_sub_att']
                    total_reversals_against += opponent_fight.iloc[0]['r_rev']
                    total_control_time_against += opponent_fight.iloc[0]['r_ctrl_sec']

        stats['avg_takedowns_against_per_fight'] = total_takedowns_against / stats['total_fights']
        stats['avg_submission_attempts_against_per_fight'] = total_submissions_against / stats['total_fights']
        stats['avg_reversals_against_per_fight'] = total_reversals_against / stats['total_fights']
        stats['avg_control_time_against_per_fight'] = total_control_time_against / stats['total_fights']

        # Knockdown statistics (with totals)
        total_knockdowns_landed = fighter_df['knockdowns'].sum()
        stats['total_knockdowns_landed'] = total_knockdowns_landed
        stats['avg_knockdowns_per_fight'] = total_knockdowns_landed / stats['total_fights']

        total_knockdowns_against = 0
        for _, fight in fighter_df.iterrows():
            if fight['is_red']:
                opponent_fight = self.df[
                    (self.df['r_fighter'] == fighter_name) &
                    (self.df['b_fighter'] == fight['opponent']) &
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    total_knockdowns_against += opponent_fight.iloc[0]['b_kd']
            else:
                opponent_fight = self.df[
                    (self.df['b_fighter'] == fighter_name) &
                    (self.df['r_fighter'] == fight['opponent']) &
                    (self.df['event_date'] == fight['date'])
                ]
                if not opponent_fight.empty:
                    total_knockdowns_against += opponent_fight.iloc[0]['r_kd']

        stats['total_knockdowns_against'] = total_knockdowns_against
        stats['avg_knockdowns_against_per_fight'] = total_knockdowns_against / stats['total_fights']

        # Calculate win rate
        stats['win_rate'] = stats['wins'] / stats['total_fights'] if stats['total_fights'] > 0 else 0

        # Calculate average fight time
        total_fight_time = fighter_df['total_time'].sum()
        stats['avg_fight_time_seconds'] = total_fight_time / stats['total_fights']

        # Store UFC fight history
        stats['fight_history'] = fighter_df.to_dict('records')

        return stats

def main():
    root = tk.Tk()
    FighterComparisonGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()