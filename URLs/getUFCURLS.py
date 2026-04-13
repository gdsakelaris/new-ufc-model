import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import re
import time

def scrape_ufc_events():
    """
    Scrapes UFC event data from ufcstats.com and outputs to Excel spreadsheet
    """
    
    # Base URL for UFC stats - we'll scrape the "all" page to get everything
    base_url = "http://ufcstats.com/statistics/events/completed?page=all"
    
    # Headers to mimic a real browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print("Fetching UFC events data...")
    
    try:
        # Make the request
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the events table
        events_table = soup.find('table', class_='b-statistics__table-events')
        
        if not events_table:
            print("Could not find events table on the page")
            return
        
        # Find all event rows (skip header and empty rows)
        event_rows = events_table.find('tbody').find_all('tr', class_='b-statistics__table-row')
        
        events_data = []
        
        for row in event_rows:
            # Skip empty rows
            if 'b-statistics__table-row_type_first' in row.get('class', []) or len(row.find_all('td')) < 2:
                continue
                
            # Extract event data from each row
            cells = row.find_all('td')
            
            if len(cells) >= 2:
                # First cell contains event name and date
                first_cell = cells[0]
                
                # Find the event link and name
                event_link = first_cell.find('a')
                if event_link:
                    event_name = event_link.get_text(strip=True)
                    event_url = event_link.get('href', '')
                    
                    # Find the date
                    date_span = first_cell.find('span', class_='b-statistics__date')
                    if date_span:
                        date_text = date_span.get_text(strip=True)
                        
                        # Parse the date (format: "Month DD, YYYY")
                        try:
                            event_date = datetime.strptime(date_text, "%B %d, %Y")
                        except ValueError:
                            try:
                                # Try alternative format
                                event_date = datetime.strptime(date_text, "%B %d, %Y")
                            except ValueError:
                                print(f"Could not parse date: {date_text}")
                                event_date = None
                        
                        # Second cell contains location (optional for now)
                        location = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                        
                        # Add to our data list
                        events_data.append({
                            'Event Date': event_date,
                            'Event Name': event_name,
                            'Event URL': event_url,
                            'Location': location  # Extra column for reference
                        })
                        
                        print(f"Extracted: {event_name} - {date_text}")
        
        print(f"\nTotal events extracted: {len(events_data)}")
        
        # Create DataFrame
        df = pd.DataFrame(events_data)
        
        # Sort by date (newest first to match the website)
        df = df.sort_values('Event Date', ascending=False)
        
        # Format the date column for Excel
        df['Event Date'] = df['Event Date'].dt.strftime('%m/%d/%y')
        
        # Create the final DataFrame with just the required columns
        final_df = df[['Event Date', 'Event Name', 'Event URL']].copy()
        
        # Save to Excel with formatting similar to the provided file
        output_filename = 'UFC_Events_URLs.xlsx'
        
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            # Write to main sheet
            final_df.to_excel(writer, sheet_name='All Events', index=False)
            
            # Get the workbook and worksheet for formatting
            workbook = writer.book
            worksheet = writer.sheets['All Events']
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"\nData saved to: {output_filename}")
        print(f"Total events saved: {len(final_df)}")
        
        # Display a sample of the data
        print("\nSample of extracted data:")
        print(final_df.head(10).to_string(index=False))
        
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
    except Exception as e:
        print(f"Error processing data: {e}")

def scrape_multiple_pages():
    """
    Alternative function to scrape multiple pages if the 'all' page doesn't work
    """
    print("Attempting to scrape multiple pages...")
    
    base_url = "http://ufcstats.com/statistics/events/completed"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    all_events = []
    page = 1
    max_pages = 30  # Safety limit
    
    while page <= max_pages:
        url = f"{base_url}?page={page}" if page > 1 else base_url
        print(f"Scraping page {page}...")
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            events_table = soup.find('table', class_='b-statistics__table-events')
            
            if not events_table:
                print(f"No events table found on page {page}")
                break
            
            event_rows = events_table.find('tbody').find_all('tr', class_='b-statistics__table-row')
            
            page_events = 0
            for row in event_rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    first_cell = cells[0]
                    event_link = first_cell.find('a')
                    
                    if event_link:
                        event_name = event_link.get_text(strip=True)
                        event_url = event_link.get('href', '')
                        
                        date_span = first_cell.find('span', class_='b-statistics__date')
                        if date_span:
                            date_text = date_span.get_text(strip=True)
                            
                            try:
                                event_date = datetime.strptime(date_text, "%B %d, %Y")
                            except ValueError:
                                event_date = None
                            
                            location = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                            
                            all_events.append({
                                'Event Date': event_date,
                                'Event Name': event_name,
                                'Event URL': event_url,
                                'Location': location
                            })
                            page_events += 1
            
            if page_events == 0:
                print(f"No events found on page {page}, stopping.")
                break
                
            print(f"Found {page_events} events on page {page}")
            
            # Check if there's a next page
            pagination = soup.find('ul', class_='b-statistics__paginate')
            if pagination:
                next_page_link = pagination.find('a', href=re.compile(f'page={page + 1}'))
                if not next_page_link:
                    print("No more pages found.")
                    break
            
            page += 1
            time.sleep(1)  # Be respectful to the server
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break
    
    print(f"Total events collected: {len(all_events)}")
    
    # Process and save the data
    if all_events:
        df = pd.DataFrame(all_events)
        df = df.sort_values('Event Date', ascending=False)
        df['Event Date'] = df['Event Date'].dt.strftime('%m/%d/%y')
        
        final_df = df[['Event Date', 'Event Name', 'Event URL']].copy()
        
        output_filename = 'UFC_Events_URLs_Multi_Page.xlsx'
        final_df.to_excel(output_filename, index=False)
        
        print(f"Data saved to: {output_filename}")
        print(f"Total events saved: {len(final_df)}")

if __name__ == "__main__":
    print("UFC Event Data Scraper")
    print("=" * 50)
    
    # Try the main scraping function first
    print("Attempting to scrape all events from single page...")
    scrape_ufc_events()
    
    # Uncomment the line below if you want to try the multi-page approach
    # scrape_multiple_pages()