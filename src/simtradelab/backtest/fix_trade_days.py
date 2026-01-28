
import pandas as pd
import os

DATA_PATH = '/mnt/c/QMTReal/SimTrade/SimTradeLab/data/ptrade_data.h5'

def fix_trade_days():
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
        return

    try:
        store = pd.HDFStore(DATA_PATH, 'a')
        
        if '/trade_days' in store.keys():
            print("/trade_days already exists.")
            # Check if it's readable
            try:
                df = store['/trade_days']
                print(f"Existing trade_days: {len(df)}")
            except Exception as e:
                print(f"Error reading existing trade_days: {e}")
        
        if '/benchmark' not in store.keys():
            print("/benchmark not found, cannot recover trade_days.")
            store.close()
            return

        print("Reading /benchmark...")
        benchmark_df = store['/benchmark']
        trade_days = benchmark_df.index
        
        print(f"Found {len(trade_days)} trading days from benchmark.")
        
        # Create DataFrame for trade_days with a dummy column
        trade_days_df = pd.DataFrame({'is_trading': 1}, index=trade_days)
        
        # Force delete if exists to ensure clean write
        if '/trade_days' in store.keys():
            print("Removing existing /trade_days...")
            store.remove('/trade_days')

        print("Writing /trade_days...")
        store.put('/trade_days', trade_days_df, format='table', complib='blosc', complevel=9)
        store.flush()
        store.close()
        
        # Re-open to verify
        print("Re-opening store to verify...")
        store = pd.HDFStore(DATA_PATH, 'r')
        if '/trade_days' in store.keys():
             print(f"Verification successful: /trade_days exists with {len(store['/trade_days'])} rows.")
        else:
             print("Verification FAILED: /trade_days not found after writing!")
        
    except Exception as e:
        print(f"Error: {e}")
        if 'store' in locals() and store.is_open:
            store.close()

if __name__ == "__main__":
    fix_trade_days()
