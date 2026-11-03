import pandas as pd
import wrds

class WRDSConnection:
    def __init__(self, username):
        self.username = username
        self.wrds = wrds.Connection(wrds_username=self.username)
    
    def libraries(self):
        return self.wrds.list_libraries()

    def tables(self, library):
        return self.wrds.list_tables(library)

    
    
    def get_agg_holdings(self, report_date='2026-06-30'):
        agg_query = """
            SELECT * FROM crsp.fund_names WHERE ticker = 'AGG'
        """ 
        agg_wrds_df = self.wrds.raw_sql(agg_query)
        agg_wrds_df.sort_values(by='chgdt', ascending=False, inplace=True) 

        portno = int(agg_wrds_df.sort_values("chgdt")["crsp_portno"].iloc[-1])  # most recent AGG portfolio
        col_to_keep = ['security_name', 'cusip', 'crsp_company_key', 'ticker', 'coupon', 'maturity_dt', 'nbr_shares', 'market_val', 'percent_tna','security_rank']
        
        holdings_query = f"""
            SELECT {', '.join(col_to_keep)} FROM crsp.holdings
            WHERE crsp_portno = {portno}
              AND report_dt = '{report_date}'
        """
        holdings_df = self.wrds.raw_sql(holdings_query)
        return holdings_df


