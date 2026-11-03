def krd(date, key_rates, ttm, coupon): 
    """
    Calculate the Key Rate Duration (KRD) for a bond given its time to maturity (ttm), coupon rate.
    Assumes face value of $100.
    
    Parameters:
    key_rates (list): List of key maturities
    ttm (float): Time to maturity of the bond in years.
    coupon (float): Annual coupon rate of the bond as a decimal (e.g., 0.05 for 5%).
    yield_curve (dictionary): Dictionary of yield per year (could also be table?) 

    Returns:
    list: List of KRD of bond based for each inputted key rate.
    """
    # Calculate the present value of cash flows
    yield_curve = get_yield_curve(date)
    durations = []
    for k in range(len(key_rates)): 
        new_rates = key_rates.copy()
        new_rates[k] += 0.0001
        pv = 0
        for t in range(1, int(ttm) + 1):
            # add feature for intrayear maturity -- interpolate on the yield curve
            cash_flow = coupon if t < ttm else coupon + 100  # Final payment includes face value
            pv += cash_flow / ((1 + yield_curve[t]) ** t)
        durations.append(pv)

    # Calculate the weighted average time to receive cash flows
    weighted_times = [(t * pv) / sum(durations) for t, pv in enumerate(durations, start=1)]
    
    # Key Rate Duration is the time-weighted sum of all durations
    krd = sum(weighted_times)
    
    return krd

def get_krd(bonds, key_rates, date): 
    bonds['krd'] = bonds.apply(lambda row: krd(date = date,
                                               key_rates = key_rates,
                                               ttm = row['TTM'],
                                               coupon = row['coupon']))

def get_yield_curve(date):
    pass 

def get_bond_price(date, ttm, coupon):
    yield_curve = get_yield_curve(date)
    pv = 0
    for t in range(1, int(ttm) + 1):
        cash_flow = coupon if t < ttm else coupon + 100  # Final payment includes face value
        pv += cash_flow / ((1 + yield_curve[t]) ** t)
    return pv
    