from data_loader import FinancialDataLoader
from cash_flow import CashFlowEngine
import pandas as pd

loader = FinancialDataLoader()
loader.resolve_missing_amounts() 

engine = CashFlowEngine(loader)

# Run the test against a few requests
test_requests = ['request_27', 'request_28', 'request_29']
results = []

print("\n--- Financial Reasoning Engine Output ---")
for req_id in test_requests:
    res = engine.evaluate_request(req_id)
    results.append(res)
    
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))