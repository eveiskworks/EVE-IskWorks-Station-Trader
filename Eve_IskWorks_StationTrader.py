import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import requests
import pandas as pd
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

#R u seriously going through my code? 
#Like Why?
ESI_BASE = "https://esi.evetech.net/latest"
# Predefined Stations with their IDs and Region IDs
STATIONS = {
    "Jita 4-4 (The Forge)": {"station_id": 60003760, "region_id": 10000002},
    "Amarr VIII (Domain)": {"station_id": 60008494, "region_id": 10000043},
    "Dodixie IX (Sinq Laison)": {"station_id": 60011866, "region_id": 10000030},
    "Hek VIII (Metropolis)": {"station_id": 60005686, "region_id": 10000042}
}

# Base names for station-specific cache files all will have station ID appended
ITEM_CACHE_FILE = "item_names.json"
VOLUME_CACHE_BASE = "daily_volume_cache.json"
ORDERS_CACHE_BASE = "market_orders_cache.json" # Base name for raw market orders cache

ITEMS_PER_PAGE = 100
MAX_CANDIDATES_FOR_VOLUME = 100

# Cache Expiration Settings
#Change these to adjust cache durations the 7 is for 7 days and 12 is for 12 hours
CACHE_EXPIRY_7_DAYS_SECONDS = 7 * 24 * 60 * 60  # For initial comprehensive volume check
CACHE_EXPIRY_12_HOURS_SECONDS = 12 * 60 * 60    # For orders and top 100 priority refresh (12 hours)

# Global State for Station
current_station_name = list(STATIONS.keys())[0]
current_station_id = STATIONS[current_station_name]["station_id"]
current_region_id = STATIONS[current_station_name]["region_id"]

# Global Data Stores
market_orders_df = pd.DataFrame()           # Stores all raw market orders fetched from ESI
average_prices_map = {}                     # Stores EVE-wide average prices (from /markets/prices)
daily_volume_map = {}                       # Stores filtered daily transaction volume (type_id: volume)
daily_volume_cache_with_timestamp = {}      # Stores {type_id: {'volume': V, 'timestamp': T}}

# GUI References (initialized in create_gui)
tree = None
root = None
status_label = None
progress_var = None
page_label = None
prev_button = None
next_button = None
station_var = None
station_combo = None

# Input Entry References (initialized in create_gui)
buy_brokrage_entry, sell_brokrage_entry, sales_tax_entry, scc_surcharge_entry = None, None, None, None
min_profit_entry, max_cost_entry, max_profit_entry, max_roi_entry, min_vol_entry, search_entry = None, None, None, None, None, None

current_page = 1
max_pages = 1
sorted_df = pd.DataFrame()
current_sort_column = 'Total Potential Profit (ISK)'
current_sort_direction = False # So as to Sort in Descenting order by default

# --- UTILITY & CACHING FUNCTIONS ---
def get_station_specific_filepath(base_filename):
    """Generates a dynamic file path based on the current station ID."""
    global current_station_id
    name, ext = os.path.splitext(base_filename)
    return f"{name}_{current_station_id}{ext}"

def load_item_names():
    """Loads item ID to name mapping from a local JSON cache."""
    if os.path.exists(ITEM_CACHE_FILE):
        try:
            with open(ITEM_CACHE_FILE, "r") as f:
                return {int(k): v for k, v in json.load(f).items()}
        except Exception as e:
            print(f"Error loading item name cache: {e}")
            return {}
    return {}

def save_item_names(item_names):
    """Saves item ID to name mapping to the local JSON cache."""
    with open(ITEM_CACHE_FILE, "w") as f:
        json.dump({str(k): v for k, v in item_names.items()}, f)

def load_volume_cache():
    """Loads daily transaction volume from the station-specific JSON cache, including timestamps."""
    global daily_volume_cache_with_timestamp, current_station_id
    daily_volume_cache_with_timestamp = {}
    volume_filepath = get_station_specific_filepath(VOLUME_CACHE_BASE)
    if os.path.exists(volume_filepath):
        try:
            with open(volume_filepath, "r") as f:
                cached_data = json.load(f)
                daily_volume_cache_with_timestamp = {
                    int(k): v for k, v in cached_data.items()
                }
            print(f"Loaded volume cache for station {current_station_id} from {volume_filepath}.")
        except Exception as e:
            print(f"Error loading volume cache for {current_station_id}: {e}")
    else:
        print(f"No volume cache found for station {current_station_id}.")

def save_volume_cache():
    """Saves daily transaction volume to the station-specific JSON cache with a timestamp."""
    global daily_volume_cache_with_timestamp, current_station_id
    volume_filepath = get_station_specific_filepath(VOLUME_CACHE_BASE)
    serializable_data = {
        str(k): v for k, v in daily_volume_cache_with_timestamp.items()
    }
    with open(volume_filepath, "w") as f:
        json.dump(serializable_data, f, indent=4)

def update_daily_volume_map_from_cache(max_age_seconds):
    """Derives the active daily_volume_map (volume only) from the timestamped cache, applying a final expiry filter."""
    global daily_volume_map, daily_volume_cache_with_timestamp
    current_time = time.time()
    daily_volume_map = {}
    # We load volumes into the active map only if they are not older than max_age_seconds (7 days)
    for tid, data in daily_volume_cache_with_timestamp.items():
        if (current_time - data.get('timestamp', 0)) < max_age_seconds:
            daily_volume_map[tid] = data.get('volume', 0)

def fetch_historical_volume(type_ids_to_check, max_age_seconds, status_prefix=""):
    """
    Fetches the latest daily transaction volume for a list of type IDs.
    """
    global daily_volume_cache_with_timestamp, current_region_id
    current_time = time.time()
    new_volumes_to_fetch = []

    # Identify missing/expired IDs based on max_age_seconds
    for tid in type_ids_to_check:
        cache_entry = daily_volume_cache_with_timestamp.get(tid)
        is_expired = True
        if cache_entry and 'timestamp' in cache_entry:
            if (current_time - cache_entry['timestamp']) < max_age_seconds:
                is_expired = False
        if is_expired:
            new_volumes_to_fetch.append(tid)

    if not new_volumes_to_fetch:
        if progress_var and progress_var.get() == 0:
            status_label.config(text=f"{status_prefix} All required volumes found in fresh cache.", bootstyle=INFO)
        return

    status_label.config(text=f"{status_prefix} Fetching daily volume for {len(new_volumes_to_fetch)} items (out of {len(type_ids_to_check)} candidates)...", bootstyle=INFO)
    root.update_idletasks()

    fetched_volumes_data = {}

    # Fetch missing/expired volumes concurrently
    with ThreadPoolExecutor(max_workers=15) as executor:
        def get_volume(type_id):
            """Internal worker function to call ESI history endpoint."""
            url = f"{ESI_BASE}/markets/{current_region_id}/history/?type_id={type_id}"
            try:
                res = requests.get(url, headers={'User-Agent': 'EVE-STATION-TRADER-OS-TOOL'})
                res.raise_for_status()
                history = res.json()
                if history:
                    # Volume from the latest day in the history
                    latest_volume = history[-1]['volume']
                    return type_id, latest_volume
                return type_id, 0
            except requests.exceptions.RequestException:
                return type_id, 0

        future_to_id = {executor.submit(get_volume, type_id): type_id for type_id in new_volumes_to_fetch}
        for i, future in enumerate(future_to_id):
            type_id = future_to_id[future]
            try:
                type_id, volume = future.result()
                # Store volume with current timestamp
                fetched_volumes_data[type_id] = {'volume': volume, 'timestamp': time.time()}

                # Update progress bar
                progress_var.set(int(((i + 1) / len(new_volumes_to_fetch)) * 100))
                status_label.config(text=f"{status_prefix} Fetching daily volume: {i + 1}/{len(new_volumes_to_fetch)} items updated...", bootstyle=INFO)
                root.update_idletasks()
            except Exception:
                pass

    # Update global cache and save
    daily_volume_cache_with_timestamp.update(fetched_volumes_data)
    save_volume_cache()

# --- MARKET ORDERS CACHING AND FETCHING ---

def load_market_orders_cache():
    """Loads market orders and timestamp from the station-specific cache."""
    cache_filepath = get_station_specific_filepath(ORDERS_CACHE_BASE)
    if os.path.exists(cache_filepath):
        try:
            with open(cache_filepath, "r") as f:
                data = json.load(f)
                return data.get('orders', []), data.get('timestamp', 0)
        except Exception as e:
            print(f"Error loading market orders cache: {e}")
    return [], 0

def save_market_orders_cache(orders):
    """Saves raw market orders and current timestamp to the station-specific cache."""
    cache_filepath = get_station_specific_filepath(ORDERS_CACHE_BASE)
    cache_data = {
        'timestamp': time.time(),
        'orders': orders
    }
    try:
        with open(cache_filepath, "w") as f:
            json.dump(cache_data, f)
        print(f"Saved market orders cache for station {current_station_id}.")
    except Exception as e:
        print(f"Error saving market orders cache: {e}")

def fetch_orders_page(region_id, page_num, max_retries=3):
    """Fetches a single page of market orders from ESI with retries."""
    url = f"{ESI_BASE}/markets/{region_id}/orders/?order_type=all&page={page_num}"
    for attempt in range(max_retries):
        try:
            res = requests.get(url, headers={'User-Agent': 'EVE-STATION-TRADER-OS-TOOL'}, timeout=15)
            res.raise_for_status()
            return res.json(), int(res.headers.get('X-Pages', 1))
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                print(f"Failed to fetch page {page_num} after {max_retries} attempts.")
                return [], 1
    return [], 1

def fetch_all_orders_concurrent(region_id):
    """Fetches all market order pages concurrently using threads for a given region."""
    status_label.config(text=f"1/3: Determining total market order pages for region {region_id}...", bootstyle=INFO)
    root.update_idletasks()

    first_page_data, total_pages = fetch_orders_page(region_id, 1)
    if total_pages <= 0 or not first_page_data:
        return pd.DataFrame()

    all_orders = first_page_data
    pages_to_fetch = list(range(2, total_pages + 1))

    status_label.config(text=f"1/3: Total pages: {total_pages}. Starting concurrent fetch...", bootstyle=INFO)
    root.update_idletasks()

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_page = {executor.submit(fetch_orders_page, region_id, page): page for page in pages_to_fetch}
        fetched_count = 1
        for future in future_to_page:
            page = future_to_page[future]
            try:
                page_data, _ = future.result()
                all_orders.extend(page_data)
                fetched_count += 1
                progress_var.set(int((fetched_count / total_pages) * 100))
                status_label.config(text=f"1/3: Fetched {fetched_count}/{total_pages} market order pages...", bootstyle=INFO)
                root.update_idletasks()
            except Exception as e:
                print(f"Page {page} generated an exception: {e}")

    save_market_orders_cache(all_orders)

    if not all_orders:
        return pd.DataFrame()
    return pd.DataFrame(all_orders)

def fetch_average_prices():
    """Fetches EVE-wide official average prices for stability checks."""
    global average_prices_map
    url = f"{ESI_BASE}/markets/prices/"
    try:
        status_label.config(text="2/3: Fetching official universe average prices...", bootstyle=INFO)
        root.update_idletasks()
        res = requests.get(url, headers={'User-Agent': 'EVE-STATION-TRADER-OS-TOOL'})
        res.raise_for_status()
        average_prices_map = {
            item['type_id']: item['average_price'] for item in res.json() if 'average_price' in item
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching average prices: {e}")
        status_label.config(text="Error fetching average prices. Trading stability check disabled.", bootstyle=WARNING)
        average_prices_map = {}

def fetch_dynamic_data():
    """Master function to fetch all volatile market data (orders and prices) for the selected station."""
    global market_orders_df, current_region_id, current_station_id, current_station_name
    progress_var.set(0)

    # Check/Load Market Orders Cache
    all_region_orders_list, timestamp = load_market_orders_cache()
    current_time = time.time()
    
    if (current_time - timestamp) < CACHE_EXPIRY_12_HOURS_SECONDS and all_region_orders_list:
        status_label.config(text=f"1/3: Loaded market data from cache (Last updated {time.strftime('%H:%M:%S', time.localtime(timestamp))}).", bootstyle=INFO)
        all_region_orders = pd.DataFrame(all_region_orders_list)
        progress_var.set(100)
    else:
        status_label.config(text=f"Cache expired or missing. Fetching market data for region of {current_station_name}...", bootstyle=INFO)
        all_region_orders = fetch_all_orders_concurrent(current_region_id)

    try:
        if all_region_orders.empty:
            status_label.config(text=f"Failed to get market data for region {current_region_id}.", bootstyle=DANGER)
            market_orders_df = pd.DataFrame()
            return

        # Fetch EVE-wide average prices
        fetch_average_prices()

        # FILTER orders to only include those at the selected station
        market_orders_df = all_region_orders[
            all_region_orders['location_id'] == current_station_id
        ].copy()

        if market_orders_df.empty:
            status_label.config(text=f"No orders found at station {current_station_name}.", bootstyle=DANGER)
            return

        # Proceed to metric calculation
        item_names = load_item_names()
        if not item_names:
            status_label.config(text="Item names not loaded. Please click 'Update Static Data' first!", bootstyle=WARNING)
            return
            
        update_table_data(item_names, init_sort=True)

    except Exception as e:
        status_label.config(text=f"An error occurred during market fetch/processing: {e}", bootstyle=DANGER)
        messagebox.showerror("Error", f"Failed to run analysis: {e}")

# --- CORE ALGORITHM ---
def compute_metrics(orders, item_names, buy_brokrage, sell_brokrage, sales_tax, scc_surcharge, search_term):
    """ Calculates key trading metrics based on market orders."""
    required_cols = ['is_buy_order', 'type_id', 'price', 'volume_remain']
    if not all(col in orders.columns for col in required_cols):
        return pd.DataFrame()

    buy_orders = orders[orders['is_buy_order']].copy()
    sell_orders = orders[~orders['is_buy_order']].copy()

    # Find the best price for each item (Max Buy, Min Sell)
    buy_prices = buy_orders.groupby('type_id')['price'].max().reset_index().rename(columns={'price': 'buy_price'})
    sell_prices = sell_orders.groupby('type_id')['price'].min().reset_index().rename(columns={'price': 'sell_price'})

    # Merge prices
    merged = buy_prices.set_index('type_id').join(sell_prices.set_index('type_id'), how='inner').reset_index()

    # Must have a positive margin
    merged = merged[merged['sell_price'] > merged['buy_price']]

    # Calculate available volumes at the best price
    def get_volume_at_best_price(row, is_buy):
        current_orders = buy_orders if is_buy else sell_orders
        price_col = 'buy_price' if is_buy else 'sell_price'
        return current_orders[
            (current_orders['type_id'] == row['type_id']) & (current_orders['price'] == row[price_col])
        ]['volume_remain'].sum()

    merged['available_buy_vol'] = merged.apply(lambda row: get_volume_at_best_price(row, True), axis=1)
    merged['available_sell_vol'] = merged.apply(lambda row: get_volume_at_best_price(row, False), axis=1)

    # --- BROKRAGE and PROFIT CALCULATION ---
    merged['cost'] = (merged['buy_price'] * (1 + buy_brokrage)) + (merged['buy_price'] * scc_surcharge)
    merged['revenue'] = merged['sell_price'] * (1 - sell_brokrage - sales_tax)
    merged['profit_per_item'] = merged['revenue'] - merged['cost']
    merged['roi'] = (merged['profit_per_item'] / merged['cost']) * 100

    # Map item ID to readable name
    merged['type_name'] = merged['type_id'].astype(str).apply(lambda tid: item_names.get(int(tid), f"ID {tid}"))

    # --- PRICE STABILITY METRICS ---
    global average_prices_map
    merged['average_price'] = merged['type_id'].apply(lambda tid: average_prices_map.get(tid, np.nan))
    merged['price_vs_avg_%'] = (
        (merged['sell_price'] - merged['average_price']) / merged['average_price']
    ) * 100
    merged['price_vs_avg_%'] = merged['price_vs_avg_%'].fillna(0)
    merged = merged.replace([np.inf, -np.inf], 0)

    # Apply search filter (if any)
    if search_term:
        merged = merged[merged['type_name'].str.contains(search_term, case=False, na=False)]

    return merged.copy()

def update_table_data(item_names, init_sort=False):
    """
    Orchestrates metric computation, two-pass volume fetch, final filtering, and preparation for display.
    """
    global market_orders_df, sorted_df, max_pages, current_page
    if init_sort:
        global current_sort_column, current_sort_direction
        current_sort_column = 'Total Potential Profit (ISK)'
        current_sort_direction = False

    if market_orders_df.empty:
        status_label.config(text=f"Error: Market data for {current_station_name} is not loaded. Please click 'Update Market Data'.", bootstyle=DANGER)
        return

    try:
        # Get user input and convert percentages to decimals
        buy_brokrage = float(buy_brokrage_entry.get()) / 100.0
        sell_brokrage = float(sell_brokrage_entry.get()) / 100.0
        sales_tax = float(sales_tax_entry.get()) / 100.0
        scc_surcharge = float(scc_surcharge_entry.get()) / 100.0
        min_vol = float(min_vol_entry.get())
        min_profit = float(min_profit_entry.get())
        max_cost = float(max_cost_entry.get())
        max_profit = float(max_profit_entry.get())
        max_roi = float(max_roi_entry.get())
        search_term = search_entry.get()
    except ValueError:
        status_label.config(text="Input fields must contain valid numbers.", bootstyle=DANGER)
        messagebox.showerror("Input Error", "Please ensure all filter fields contain valid numbers.")
        return

    status_label.config(text="1/4: Computing profitability for all market items...", bootstyle=INFO)
    root.update_idletasks()

    # Compute profitability metrics (no volume filtering yet)
    all_candidates = compute_metrics(
        market_orders_df, item_names, buy_brokrage, sell_brokrage, sales_tax, scc_surcharge, search_term
    )

    # Apply immediate cost/profit/ROI filters
    filtered_df = all_candidates[
        (all_candidates['profit_per_item'] >= min_profit) &
        (all_candidates['cost'] <= max_cost) &
        (all_candidates['profit_per_item'] <= max_profit) &
        (all_candidates['roi'] <= max_roi)
    ].copy()

    if filtered_df.empty:
        if tree:
            for row in tree.get_children():
                tree.delete(row)
        sorted_df = pd.DataFrame()
        max_pages = 1
        current_page = 1
        update_pagination_buttons()
        status_label.config(text=f"No profitable items matched the current cost/profit/ROI/search filters for {current_station_name}.", bootstyle=WARNING)
        progress_var.set(0)
        return

    all_profitable_ids = filtered_df['type_id'].unique().tolist()
    load_volume_cache() # Load the existing station-specific cache data

    # First Volume Pass: Check ALL profitable items against the 7-day cache policy (COMPREHENSIVE)
    status_label.config(text="2/4: Initial volume check for ALL profitable items (7-day policy)...", bootstyle=INFO)
    progress_var.set(0)
    root.update_idletasks()
    fetch_historical_volume(
        all_profitable_ids,
        CACHE_EXPIRY_7_DAYS_SECONDS,
        status_prefix="2a/4: "
    )

    # Second Volume Pass: Identify Top 100 and check against the 12-hour policy (PRIORITY)
    update_daily_volume_map_from_cache(CACHE_EXPIRY_7_DAYS_SECONDS) # Update map after initial fetch

    # Temporarily map volume to determine the TOP 100
    filtered_df['volume_day_txn'] = filtered_df['type_id'].apply(lambda tid: daily_volume_map.get(tid, 0))
    filtered_df['total_potential_profit'] = filtered_df['profit_per_item'] * filtered_df['volume_day_txn']

    # Sort by total profit and get the top candidates ID list
    top_100_ids = filtered_df.sort_values(
        by='total_potential_profit',
        ascending=False
    ).head(MAX_CANDIDATES_FOR_VOLUME)['type_id'].tolist()

    if top_100_ids:
        status_label.config(text=f"3/4: Priority volume check for TOP {len(top_100_ids)} items (12-hour policy)...", bootstyle=INFO)
        progress_var.set(0)
        root.update_idletasks()
        fetch_historical_volume(
            top_100_ids,
            CACHE_EXPIRY_12_HOURS_SECONDS,
            status_prefix="3b/4: "
        )

    # Final Metric Calculation and Filtering
    update_daily_volume_map_from_cache(CACHE_EXPIRY_7_DAYS_SECONDS) # Final update of the active volume map

    # Re-map volume and profit metrics to the DataFrame
    final_df = filtered_df.copy()
    final_df['volume_day_txn'] = final_df['type_id'].apply(lambda tid: daily_volume_map.get(tid, 0))
    final_df['total_potential_profit'] = final_df['profit_per_item'] * final_df['volume_day_txn']

    # Apply the final MINIMUM transaction volume filter
    final_df = final_df[final_df['volume_day_txn'] >= min_vol]

    # Prepare float column for accurate numerical sorting (Vs Avg)
    final_df['price_vs_avg_float'] = final_df['price_vs_avg_%'].copy()

    # Store the final, filtered data and refresh view
    sorted_df = final_df.copy()
    apply_sort_and_refresh()

# --- STATION LOGIC ---
def select_station(event=None):
    """Handles station selection change, updates global state, and loads caches."""
    global current_station_name, current_station_id, current_region_id, root
    selected_name = station_var.get()

    if not selected_name:
        return

    if selected_name == current_station_name and not market_orders_df.empty:
        return

    current_station_name = selected_name
    station_data = STATIONS[selected_name]
    new_station_id = station_data["station_id"]
    new_region_id = station_data["region_id"]

    # Update globals
    current_station_id = new_station_id
    current_region_id = new_region_id
    status_label.config(text=f"Switched to {current_station_name}. Loading cache...", bootstyle=PRIMARY)
    root.update_idletasks()

    # Clear old displayed results and load new station's volume cache
    clear_data_stores()
    load_volume_cache()

    # Update title
    root.title(f"EVE IskWorks ({current_station_name})") # Renamed application
    status_label.config(text=f"Ready for {current_station_name}. Press 'Update Market Data' to begin or 'Apply Filters' to see cached results.", bootstyle=INFO)

def clear_data_stores():
    """Resets global data stores when switching stations."""
    global market_orders_df, sorted_df, current_page, max_pages
    # Reset DataFrames
    market_orders_df = pd.DataFrame()
    sorted_df = pd.DataFrame()

    # Clear Treeview
    if tree:
        for item in tree.get_children():
            tree.delete(item)

    # Reset Pagination State
    current_page = 1
    max_pages = 1
    update_pagination_buttons()

# --- ESI ITEM NAME UTILITIES (Static Data) ---
def fetch_all_item_ids():
    """Fetches all item type IDs from ESI across multiple pages."""
    ids = []
    page = 1
    while True:
        try:
            res = requests.get(f"{ESI_BASE}/universe/types/?page={page}", headers={'User-Agent': 'EVE-STATION-TRADER-OS-TOOL'}, timeout=10)
            res.raise_for_status()
            batch = res.json()
            if not batch:
                break
            ids.extend(batch)
            page += 1
            time.sleep(0.05)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404: break
            else: print(f"ERROR: Unexpected HTTP error fetching item IDs page {page}: {e}"); break
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Connection error fetching item IDs page {page}: {e}"); break
    return ids

def resolve_item_names_batch(type_ids):
    """Resolves a list of item IDs to their names via ESI."""
    try:
        res = requests.post(f"{ESI_BASE}/universe/names/", json=type_ids, headers={'User-Agent': 'EVE-STATION-TRADER-OS-TOOL'})
        res.raise_for_status()
        return {item['id']: item['name'] for item in res.json() if item.get('category') == 'inventory_type'}
    except requests.exceptions.RequestException as e:
        print(f"Error resolving item names batch: {e}")
        return {}

def fetch_static_data():
    """Fetches all item IDs and resolves/caches their names."""
    status_label.config(text="1/3: Loading/Updating static item name cache...", bootstyle=WARNING)
    root.update_idletasks()
    try:
        item_names = load_item_names()
        all_ids = fetch_all_item_ids()
        # Use integer comparison for IDs since item_names keys are integers after load_item_names
        new_ids_to_resolve = [tid for tid in all_ids if tid not in item_names] 

        if new_ids_to_resolve:
            status_label.config(text=f"2/3: Resolving names for {len(new_ids_to_resolve)} new items...", bootstyle=INFO)
            for i in range(0, len(new_ids_to_resolve), 1000): # ESI limit
                batch_ids = new_ids_to_resolve[i:i+1000]
                resolved = resolve_item_names_batch(batch_ids)
                item_names.update(resolved)
                progress_var.set(int((i / len(new_ids_to_resolve)) * 100))
                root.update_idletasks()

            save_item_names(item_names)
            status_label.config(text=f"3/3: Item name cache updated with {len(item_names)} items. Ready for market fetch.", bootstyle=SUCCESS)
            progress_var.set(100)
        else:
             status_label.config(text=f"3/3: Item name cache is up to date with {len(item_names)} items. Ready for market fetch.", bootstyle=SUCCESS)
             progress_var.set(100)

    except Exception as e:
        status_label.config(text=f"Error during static data fetch: {e}", bootstyle=DANGER)
        messagebox.showerror("Error", f"Failed to fetch static data: {e}")

# --- THREADING HELPERS ---
def fetch_static_data_thread():
    """Starts static data fetch in a separate thread."""
    threading.Thread(target=fetch_static_data, daemon=True).start()

def fetch_dynamic_data_thread():
    """Starts dynamic market data fetch and analysis in a separate thread."""
    if not load_item_names():
        status_label.config(text="Please update static item names first.", bootstyle=DANGER)
        messagebox.showwarning("Prerequisite", "Please run 'Update Static Data (Item Names)' first.")
        return
    threading.Thread(target=fetch_dynamic_data, daemon=True).start()

def apply_filters_thread():
    """Starts metric calculation and final filtering based on current UI inputs."""
    if market_orders_df.empty:
        status_label.config(text=f"Please fetch market data for {current_station_name} first.", bootstyle=DANGER)
    else:
        threading.Thread(target=lambda: update_table_data(load_item_names()), daemon=True).start()

# --- GUI / DISPLAY LOGIC ---
def apply_sort_and_refresh(event=None, new_col_name=None):
    """Sorts the stored data based on the header click and refreshes the current page view."""
    global sorted_df, max_pages, current_page, current_sort_column, current_sort_direction

    if sorted_df.empty:
        update_pagination_buttons()
        return

    if new_col_name:
        if new_col_name == current_sort_column:
            current_sort_direction = not current_sort_direction
        else:
            current_sort_column = new_col_name
            current_sort_direction = False

    column_map = {
        "Item": 'type_name', "Buy (Max)": 'buy_price', "Sell (Min)": 'sell_price', "Cost (w/ Fees)": 'cost',
        "Profit/Item": 'profit_per_item', "Total Potential Profit (ISK)": 'total_potential_profit',
        "ROI (%)": 'roi', "Available Buy Vol": 'available_buy_vol', "Available Sell Vol": 'available_sell_vol',
        "Vol/Day (Txn)": 'volume_day_txn', "Avg Price": 'average_price', "Vs Avg (%)": 'price_vs_avg_float'
    }

    sort_key = column_map.get(current_sort_column, 'total_potential_profit')
    
    sorted_df = sorted_df.sort_values(by=sort_key, ascending=current_sort_direction)

    max_pages = max(1, int(np.ceil(len(sorted_df) / ITEMS_PER_PAGE)))
    current_page = 1
    refresh_treeview_page()
    status_label.config(text=f"Analysis complete for {current_station_name}. Displaying {len(sorted_df):,} items.", bootstyle=SUCCESS)

def refresh_treeview_page():
    """Updates the Treeview with the data for the current page."""
    global sorted_df, current_page

    if tree:
        for row in tree.get_children():
            tree.delete(row)

    if sorted_df.empty:
        update_pagination_buttons()
        return

    start_index = (current_page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    page_data = sorted_df.iloc[start_index:end_index]

    def format_isk(value):
        if pd.isna(value): return ""
        return f"{value:,.2f}"

    def format_volume(value):
        if pd.isna(value): return ""
        return f"{int(value):,}"

    for _, row in page_data.iterrows():
        color_tag = ''
        
        if row['price_vs_avg_%'] > 10:
            color_tag = 'high_deviation' 
        elif row['price_vs_avg_%'] < -10:
            color_tag = 'low_deviation' 

        tree.insert("", "end", tags=(color_tag,), values=(
            row['type_name'],
            format_isk(row['buy_price']),
            format_isk(row['sell_price']),
            format_isk(row['cost']),
            format_isk(row['profit_per_item']),
            format_isk(row['total_potential_profit']),
            f"{row['roi']:.2f}%",
            format_volume(row['available_buy_vol']),
            format_volume(row['available_sell_vol']),
            format_volume(row['volume_day_txn']),
            format_isk(row['average_price']),
            f"{row['price_vs_avg_%']:.1f}%"
        ))
    update_pagination_buttons()

def navigate_page(direction):
    """Changes the current page and refreshes the display."""
    global current_page
    new_page = current_page + direction
    if 1 <= new_page <= max_pages:
        current_page = new_page
        refresh_treeview_page()

def update_pagination_buttons():
    """Updates the page number display and enables/disables buttons."""
    global current_page, max_pages, page_label, prev_button, next_button
    page_label.config(text=f"Page {current_page} of {max_pages} ({len(sorted_df):,} Total Items)")
    prev_button.config(state=DISABLED if current_page == 1 else NORMAL)
    next_button.config(state=DISABLED if current_page == max_pages or max_pages == 1 else NORMAL)

def create_gui():
    """Initializes and runs the Tkinter GUI."""
    global root, tree, status_label, progress_var, page_label, prev_button, next_button
    global buy_brokrage_entry, sell_brokrage_entry, sales_tax_entry, scc_surcharge_entry, min_vol_entry, min_profit_entry, max_cost_entry, max_profit_entry, max_roi_entry, search_entry
    global station_var, station_combo

    # Initialize the modern Bootstrap window
    root = ttk.Window(themename="darkly")
    root.state('zoomed')
    root.resizable(True, True)
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    # Set the initial title here
    root.title(f"EVE IskWorks ({current_station_name})") 


    # Main Container Frame
    main_frame = ttk.Frame(root, padding=15)
    main_frame.grid(row=0, column=0, sticky="nsew")
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(3, weight=1)

    # --- TOP BUTTONS AND INFO (Row 0) ---
    top_controls = ttk.Frame(main_frame)
    top_controls.grid(column=0, row=0, sticky='ew', columnspan=2, pady=(0, 10))
    top_controls.grid_columnconfigure(0, weight=1)
    top_controls.grid_columnconfigure(1, weight=1)
    top_controls.grid_columnconfigure(2, weight=1)

    # Station Selector
    station_names = list(STATIONS.keys())
    station_var = tk.StringVar(value=station_names[0])
    station_combo = ttk.Combobox(top_controls, textvariable=station_var, values=station_names, state='readonly', width=30, bootstyle="info")
    station_combo.grid(column=0, row=0, sticky='w', padx=(0, 10))
    station_combo.bind("<<ComboboxSelected>>", select_station)

    # Update Static Data Button
    ttk.Button(top_controls, text="Update Static Data (Item Names)", command=fetch_static_data_thread, bootstyle=(WARNING, OUTLINE)).grid(column=1, row=0, padx=10)

    # Update Market Data Button
    ttk.Button(top_controls, text="Update Market Data", command=fetch_dynamic_data_thread, bootstyle=(SUCCESS, OUTLINE)).grid(column=2, row=0, sticky='e')

    # --- FILTERS/INPUTS FRAME (Row 1) ---
    input_frame = ttk.Frame(main_frame)
    input_frame.grid(column=0, row=1, sticky='ew', columnspan=2, pady=10)

    for i in range(5):
        input_frame.grid_columnconfigure(i * 2, weight=1)
        input_frame.grid_columnconfigure(i * 2 + 1, weight=1)

    inputs = [
        ("Buy Brokerage (%)", "0.00"), ("Sell Brokerage (%)", "1.00"), ("Sales Tax (%)", "3.30"),
        ("SCC Surcharge (%)", "0.50"), ("Min Profit (ISK)", "0.1"), ("Max Cost (ISK)", "999999999"),
        ("Max Profit (ISK)", "100000000"), ("Max ROI (%)", "1000"), ("Min Volume/Day (Units)", "10"),
        ("Search Item (contains)", "")
    ]

    entries = []
    for i, (label, default) in enumerate(inputs):
        col = (i % 5) * 2
        row = i // 5
        ttk.Label(input_frame, text=label + ":", bootstyle=LIGHT).grid(column=col, row=row, sticky='w', padx=(10, 2), pady=5)
        entry = ttk.Entry(input_frame, width=20, bootstyle=PRIMARY)
        entry.insert(0, default)
        entry.grid(column=col+1, row=row, sticky='ew', padx=(2, 10), pady=5)
        entries.append(entry)

    buy_brokrage_entry, sell_brokrage_entry, sales_tax_entry, scc_surcharge_entry, min_profit_entry, max_cost_entry, max_profit_entry, max_roi_entry, min_vol_entry, search_entry = entries

    # Apply Filters Button
    ttk.Button(input_frame, text="Apply Filters / Recalculate Metrics", command=apply_filters_thread, bootstyle=(PRIMARY, SOLID)).grid(column=10, row=0, columnspan=1, rowspan=2, padx=10, sticky='nsew')

    # --- STATUS/PROGRESS FRAME (Row 2) ---
    status_frame = ttk.Frame(main_frame, padding=(0, 0))
    status_frame.grid(row=2, column=0, sticky="ew", columnspan=2)
    status_frame.grid_columnconfigure(0, weight=1)
    status_label = ttk.Label(status_frame, text="Ready.", bootstyle=SUCCESS)
    status_label.grid(row=0, column=0, padx=0, pady=(5, 0), sticky="w")
    progress_var = tk.IntVar()
    progress_bar = ttk.Progressbar(status_frame, variable=progress_var, maximum=100, bootstyle=(PRIMARY, STRIPED))
    progress_bar.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")

    # --- TREEVIEW WIDGET (Row 3) ---
    tree_frame = ttk.Frame(main_frame, padding=(0, 0))
    tree_frame.grid(row=3, column=0, sticky="nsew", columnspan=2)
    tree_frame.grid_columnconfigure(0, weight=1)
    tree_frame.grid_rowconfigure(0, weight=1)

    columns = [
        "Item", "Buy (Max)", "Sell (Min)", "Cost (w/ Fees)", "Profit/Item", "Total Potential Profit (ISK)",
        "ROI (%)", "Available Buy Vol", "Available Sell Vol", "Vol/Day (Txn)", "Avg Price", "Vs Avg (%)"
    ]

    # Create the Treeview widget and assign it to the global 'tree' variable
    tree = ttk.Treeview(tree_frame, columns=columns, show="headings", bootstyle="primary")

    style = ttk.Style()
    
    HIGH_DEVIATION_BG = '#A52A2A' 
    LOW_DEVIATION_BG = '#006400'  
    WHITE_COLOR = 'white' 

    style.configure("Treeview", rowheight=25)
  
    tree.tag_configure('high_deviation', 
                       background=HIGH_DEVIATION_BG, 
                       foreground=WHITE_COLOR)
    
    tree.tag_configure('low_deviation', 
                       background=LOW_DEVIATION_BG, 
                       foreground=WHITE_COLOR)

    scrollbar = ttk.Scrollbar(tree_frame, orient=VERTICAL, command=tree.yview, bootstyle="round")
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.grid(row=0, column=1, sticky='ns')
    tree.grid(row=0, column=0, sticky="nsew")

    # Define column headings and sizes
    tree.heading("Item", text="Item", anchor=W, command=lambda: apply_sort_and_refresh(new_col_name="Item"))
    tree.column("Item", width=250, anchor=W)

    for col in columns[1:]:
        tree.heading(col, text=col, anchor=E, command=lambda c=col: apply_sort_and_refresh(new_col_name=c))
        if 'ISK' in col or 'Price' in col or 'Cost' in col or 'Profit' in col:
             tree.column(col, width=120, anchor=E)
        elif 'Vol' in col or 'Day' in col:
             tree.column(col, width=100, anchor=E)
        else:
             tree.column(col, width=70, anchor=E)

    # --- PAGINATION CONTROLS (Row 4) ---
    pagination_frame = ttk.Frame(main_frame, padding=(0, 10))
    pagination_frame.grid(row=4, column=0, columnspan=2, sticky='ew')
    pagination_frame.grid_columnconfigure(0, weight=1)
    pagination_frame.grid_columnconfigure(1, weight=0)
    pagination_frame.grid_columnconfigure(2, weight=0)

    prev_button = ttk.Button(pagination_frame, text="< Prev", command=lambda: navigate_page(-1), state=DISABLED, bootstyle=(PRIMARY, OUTLINE))
    prev_button.grid(row=0, column=1, padx=5)

    next_button = ttk.Button(pagination_frame, text="Next >", command=lambda: navigate_page(1), state=DISABLED, bootstyle=(PRIMARY, OUTLINE))
    next_button.grid(row=0, column=3, padx=5)

    page_label = ttk.Label(pagination_frame, text=f"Page 1 of 1 ({len(sorted_df):,} Total Items)", bootstyle=LIGHT)
    page_label.grid(row=0, column=0, sticky='w')

    # Initial setup call
    select_station()

    root.mainloop()

if __name__ == '__main__':
    create_gui()