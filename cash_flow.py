import pandas as pd
from datetime import datetime, timedelta

class CashFlowEngine:
    def __init__(self, data_loader):
        self.dl = data_loader

    def get_exchange_rate(self, from_currency: str, to_currency: str, date: str) -> float:
        if from_currency == to_currency:
            return 1.0
            
        rates = self.dl.exchange_rates
        rate_row = rates[(rates['from_currency'] == from_currency) & 
                         (rates['to_currency'] == to_currency) & 
                         (rates['rate_date'] == date)]
        
        if not rate_row.empty:
            return float(rate_row.iloc[0]['rate'])
            
        past_rates = rates[(rates['from_currency'] == from_currency) & 
                           (rates['to_currency'] == to_currency) & 
                           (rates['rate_date'] <= date)].sort_values(by='rate_date', ascending=False)
                           
        if not past_rates.empty:
            return float(past_rates.iloc[0]['rate'])
            
        return 1.0

    def get_baseline_forecast(self, user_id: str, request_date: str) -> dict:
        """
        Calculates daily balances for 90 days. 
        Returns a dict of { 'YYYY-MM-DD': balance } and the lowest safe dip value.
        """
        profile = self.dl.get_user_profile(user_id)
        events = self.dl.get_events_for_user(user_id)
        
        home_currency = profile['home_currency']
        current_balance = float(profile['current_available_balance'])
        min_balance = float(profile['minimum_balance_to_keep'])
        
        start_date = datetime.strptime(request_date, "%Y-%m-%d")
        
        daily_changes = { (start_date + timedelta(days=i)).strftime("%Y-%m-%d"): 0.0 for i in range(91) }
        
        for _, event in events.iterrows():
            event_date_str = str(event['event_date'])
            if event_date_str < request_date or event_date_str > (start_date + timedelta(days=90)).strftime("%Y-%m-%d"):
                continue
                
            if event.get('status') in ['cancelled', 'failed']:
                continue
                
            amount = float(event['amount'])
            if event['currency'] != home_currency:
                amount *= self.get_exchange_rate(event['currency'], home_currency, event_date_str)
                
            if event['direction'] == 'debit':
                daily_changes[event_date_str] -= amount
            elif event['direction'] == 'credit':
                daily_changes[event_date_str] += amount
                
        running_balance = current_balance
        daily_balances = {}
        
        for i in range(91):
            date_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            running_balance += daily_changes[date_str]
            daily_balances[date_str] = running_balance
            
        return {
            'daily_balances': daily_balances,
            'min_balance_required': min_balance,
            'lowest_forecast_balance': min(daily_balances.values())
        }

    def evaluate_request(self, request_id: str) -> dict:
        req = self.dl.get_request_details(request_id)
        if not req:
            return {}

        user_id = req['user_id']
        profile = self.dl.get_user_profile(user_id)
        
        # Calculate normalized requested amount
        req_amount = float(req['requested_amount'])
        
        forecast = self.get_baseline_forecast(user_id, req['request_date'])
        lowest_bal = forecast['lowest_forecast_balance']
        min_keep = forecast['min_balance_required']
        
        # The absolute maximum we can pay today without dipping below min_balance at ANY point in the next 90 days.
        max_safe_lump_sum = max(0.0, lowest_bal - min_keep)
        
        # Cap amount_safe_to_pay at the requested amount
        amount_safe_to_pay = min(req_amount, max_safe_lump_sum)
        
        # Check earliest date for full payment
        earliest_full_date = ""
        for date_str, bal in forecast['daily_balances'].items():
            if bal - req_amount >= min_keep:
                # Check if this holds true for all subsequent days
                subsequent_bals = [b for d, b in forecast['daily_balances'].items() if d >= date_str]
                if min(subsequent_bals) - req_amount >= min_keep:
                    earliest_full_date = date_str
                    break
        
        # Initial Decision Logic
        if earliest_full_date == req['request_date']:
            status = "affordable_now"
            method = "full_payment"
            plan = f"{req['request_date']}:{req_amount}"
        elif earliest_full_date != "" and earliest_full_date <= req['desired_completion_date']:
            status = "affordable_later"
            method = "wait"
            plan = f"{earliest_full_date}:{req_amount}"
        else:
            status = "not_affordable"
            method = "not_recommended"
            plan = "none"

        return {
            "request_id": request_id,
            "amount_safe_to_pay": amount_safe_to_pay,
            "affordability_status": status,
            "recommended_payment_method": method,
            "payment_plan": plan,
            "earliest_date_for_full_payment": earliest_full_date,
            "spending_changes_needed": "none",
            "decision_explanation": f"Forecast lowest balance is {lowest_bal:.2f} against minimum {min_keep:.2f}."
        }